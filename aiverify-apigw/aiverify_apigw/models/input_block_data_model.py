from datetime import datetime

from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, validates, relationship
from typing import Optional

from .base_model import BaseORMModel
from .user_model import UserModel
from .plugin_model import InputBlockModel
from ..lib.validators import validate_gid_cid


class InputBlockDataModel(BaseORMModel):
    __tablename__ = "inputblock_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    gid: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    cid: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    inputblock_id: Mapped[int] = mapped_column(ForeignKey("inputblock.id"))
    # group: Mapped[str] = mapped_column(String(128), nullable=True)
    group_id: Mapped[Optional[int]] = mapped_column(ForeignKey("inputblock_group_data.id"))
    group: Mapped[Optional["InputBlockGroupDataModel"]] = relationship(back_populates="input_blocks")
    group_number: Mapped[Optional[int]] = mapped_column(default=0)
    inputblock: Mapped["InputBlockModel"] = relationship()
    data: Mapped[bytes]  # serialized json, output from input block save
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("user.id"))
    user: Mapped[Optional["UserModel"]] = relationship()
    created_at: Mapped[Optional[datetime]]
    updated_at: Mapped[Optional[datetime]]

    @validates("gid", "cid")
    def my_validate_gid_cid(self, key, value):
        if not validate_gid_cid(value):
            raise ValueError("Invalid GID or CID")
        return value

    def __repr__(self) -> str:
        return f"InputBlockDataModel(id={self.id}, name={self.name}, gid={self.gid}, cid={self.cid}, output={self.data.decode('utf-8')})"


class InputBlockGroupDataModel(BaseORMModel):
    __tablename__ = "inputblock_group_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gid: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    group: Mapped[str] = mapped_column(String(128), nullable=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    input_blocks: Mapped[list["InputBlockDataModel"]] = relationship(back_populates="group", cascade="all, delete")
    created_at: Mapped[Optional[datetime]]
    updated_at: Mapped[Optional[datetime]]

    def __repr__(self) -> str:
        return f"InputBlockGroupDataModel(id={self.id}, name={self.name}, gid={self.gid})"
