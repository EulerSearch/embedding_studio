from typing import Any

from fastapi import APIRouter, status

router = APIRouter()


@router.get("/ping", status_code=status.HTTP_200_OK)
def ping() -> Any:
    """Health check endpoint."""
    return {}
