"""merge multiple heads

Revision ID: b04a02947f80
Revises: 61dd7436e16c
Create Date: 2025-07-03 19:38:07.641327

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b04a02947f80'
down_revision: Union[str, None] = '61dd7436e16c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
