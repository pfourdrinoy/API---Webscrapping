import uvicorn
from fastapi.responses import RedirectResponse
from src.api.routes.data import router

from src.app import get_application

app = get_application()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
