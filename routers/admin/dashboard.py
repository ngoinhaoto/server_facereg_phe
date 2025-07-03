from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.db import get_db
from schemas.user import UserResponse
from security.auth import get_current_active_user
from models.database import Class, ClassSession, Attendance, User
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dateutil.relativedelta import relativedelta  # Add this import

router = APIRouter()

@router.get("/dashboard")
async def get_dashboard_data(
    startDate: str = None,
    endDate: str = None,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get all dashboard data in a single request"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Convert string dates to datetime objects
    try:
        start_date = datetime.fromisoformat(startDate.replace('Z', '+00:00')) if startDate else None
        end_date = datetime.fromisoformat(endDate.replace('Z', '+00:00')) if endDate else None
    except Exception as e:
        print(f"Error parsing dates: {str(e)}")
        start_date = datetime.now() - timedelta(days=150)  # Default to past 5 months
        end_date = datetime.now()
    
    # Get users data
    users = db.query(User).all()
    users_data = {
        "total": len(users),
        "admins": len([u for u in users if u.role == "admin"]),
        "teachers": len([u for u in users if u.role == "teacher"]),
        "students": len([u for u in users if u.role == "student"])
    }
    
    # Get classes with students
    classes = db.query(Class).all()
    classes_with_students = []
    
    for cls in classes:
        students_count = len(cls.students) if cls.students else 0
        classes_with_students.append({
            "id": cls.id,
            "name": cls.name,
            "class_code": cls.class_code,
            "students_count": students_count
        })
    
    # Get classes with sizes for top classes
    classes_with_sizes = []
    for cls in classes:
        classes_with_sizes.append({
            "id": cls.id,
            "name": cls.name or "Unnamed Class",
            "students": len(cls.students) if cls.students else 0
        })
    
    classes_with_sizes.sort(key=lambda x: x["students"], reverse=True)
    classes_with_sizes = classes_with_sizes[:5]  # Top 5 classes
    
    # Get attendance data for the activity chart
    attendance_data = {}
    activity_data = []
    
    # Generate months for the date range
    months = []
    current = start_date
    while current <= end_date:
        month_key = current.strftime("%Y-%m")
        months.append({
            "month_key": month_key,
            "date": current.strftime("%b"),
            "year": current.strftime("%Y")
        })
        # Move to next month - FIX: Use relativedelta instead of replace
        current = current + relativedelta(months=1)
    
    # Initialize data for each month
    for month in months:
        attendance_data[month["month_key"]] = {
            "present": 0, 
            "late": 0, 
            "absent": 0,
            "date": month["date"],
            "year": month["year"]
        }
    
    # Process actual attendance data
    for cls in classes:
        sessions = db.query(ClassSession).filter(ClassSession.class_id == cls.id).all()
        
        for session in sessions:
            session_date = session.session_date
            if start_date and session_date < start_date:
                continue
            if end_date and session_date > end_date:
                continue
                
            attendance_records = session.attendances
            month_key = session_date.strftime("%Y-%m")
            
            if month_key in attendance_data:
                for record in attendance_records:
                    status = record.status.lower()
                    if status in ["present", "late", "absent"]:
                        attendance_data[month_key][status] += 1
    
    for month_key, data in attendance_data.items():
        students_count = len([u for u in users if u.role == "student"])
        
        activity_data.append({
            "month": month_key,
            "date": data["date"],
            "year": data["year"],
            "present": data["present"],
            "late": data["late"],
            "absent": data["absent"],
            "total": data["present"] + data["late"] + data["absent"],
            "students": students_count,
            "attendance": data["present"] + data["late"]
        })
    
    activity_data.sort(key=lambda x: x["month"])

    return {
        "users": users_data,
        "classes": classes_with_students,
        "classesWithSizes": classes_with_sizes,
        "activityData": activity_data
    }