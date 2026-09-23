"""Aggregate router mounted under ``/v1``."""

from fastapi import APIRouter

from app.routes import agent, approvals, auth, knowledge, ops, organizations, tickets

router = APIRouter(prefix="/v1")

router.include_router(auth.router)
router.include_router(organizations.router)
router.include_router(knowledge.router)
router.include_router(agent.router)
router.include_router(approvals.router)
router.include_router(tickets.router)
router.include_router(ops.router)

__all__ = ["router"]
