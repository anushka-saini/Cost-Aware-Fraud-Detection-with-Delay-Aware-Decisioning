# Entry point kept for the original start command:
#
#     cd api
#     uvicorn main:app --reload
#
# The service itself lives in src/fraud/service/ (app factory, routes, database,
# auth). It can also be started from anywhere with:
#
#     uvicorn fraud.service.app:app

from fraud.service.app import app  # noqa: F401
