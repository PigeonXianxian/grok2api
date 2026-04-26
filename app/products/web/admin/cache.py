"""Local media cache management — stats, list, clear, delete."""

import asyncio
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.platform.errors import AppError, ErrorKind
from app.platform.storage import (
    clear_local_media_files,
    delete_local_media_file,
    list_local_media_files,
    local_media_stats,
)

router = APIRouter(prefix="/cache", tags=["Admin - Cache"])


class ClearCacheRequest(BaseModel):
    type: Literal["image", "video"] = "image"


class DeleteCacheItemRequest(BaseModel):
    type: Literal["image", "video"] = "image"
    name: str


class DeleteCacheItemsRequest(BaseModel):
    type: Literal["image", "video"] = "image"
    names: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("")
async def cache_stats():
    return {
        "local_image": local_media_stats("image"),
        "local_video": local_media_stats("video"),
    }


@router.get("/list")
async def list_local(
    cache_type: Literal["image", "video"] = "image",
    type_: Literal["image", "video"] | None = Query(default=None, alias="type"),
    page: int = 1,
    page_size: int = 1000,
):
    media_type = type_ or cache_type
    return {
        "status": "success",
        **list_local_media_files(media_type, page=page, page_size=page_size),
    }


@router.post("/clear")
async def clear_local(req: ClearCacheRequest):
    removed = await asyncio.to_thread(clear_local_media_files, req.type)
    return {"status": "success", "result": {"removed": removed}}


@router.post("/item/delete")
async def delete_local_item(req: DeleteCacheItemRequest):
    if not req.name:
        raise AppError(
            "Missing file name",
            kind=ErrorKind.VALIDATION,
            code="missing_file_name",
            status=400,
        )
    try:
        deleted = await asyncio.to_thread(delete_local_media_file, req.type, req.name)
    except ValueError as exc:
        raise AppError(
            str(exc),
            kind=ErrorKind.VALIDATION,
            code="invalid_file_name",
            status=400,
        ) from exc
    if not deleted:
        raise AppError(
            "File not found",
            kind=ErrorKind.VALIDATION,
            code="file_not_found",
            status=404,
        )
    return {"status": "success", "result": {"deleted": req.name}}


@router.post("/items/delete")
async def delete_local_items(req: DeleteCacheItemsRequest):
    names = [name.strip() for name in req.names if name and name.strip()]
    if not names:
        raise AppError(
            "Missing file names",
            kind=ErrorKind.VALIDATION,
            code="missing_file_names",
            status=400,
        )

    deleted = 0
    missing = 0

    for name in names:
        try:
            removed = await asyncio.to_thread(delete_local_media_file, req.type, name)
        except ValueError:
            missing += 1
            continue
        if removed:
            deleted += 1
        else:
            missing += 1

    return {
        "status": "success",
        "result": {
            "deleted": deleted,
            "missing": missing,
        },
    }
