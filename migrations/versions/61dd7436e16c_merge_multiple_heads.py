"""merge_multiple_heads

Revision ID: 61dd7436e16c
Revises: add_phe_embedding_columns, d42e492a79a6
Create Date: 2025-06-30 19:21:30.289964

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '61dd7436e16c'
down_revision: Union[str, None] = 'd42e492a79a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
