import asyncio
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.graph import Graph
from backend.services.mongodb import MongoDBService
from backend.services.pdf_service import PDFService
from backend.services.websocket_manager import WebSocketManager

# Load environment variables from .env file at startup
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

app = FastAPI(title="Tavily Company Research API")

# STATIC FILES CONFIGURATION - MUST BE BEFORE MIDDLEWARE
build_dir = Path(__file__).parent / "dist"

# Debug: Log the paths being used
logger.info(f"Build directory path: {build_dir}")
logger.info(f"Build directory absolute path: {build_dir.absolute()}")
logger.info(f"Build directory exists: {build_dir.exists()}")

if build_dir.exists():
    logger.info(f"Contents of build directory: {list(build_dir.iterdir())}")
    
    # Mount assets directory if it exists
    assets_dir = build_dir / "assets"
    if assets_dir.exists():
        logger.info(f"Mounting assets from: {assets_dir}")
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    else:
        logger.warning(f"Assets directory not found at: {assets_dir}")
    
    # Mount other static files (favicon, manifest.json, etc.) at root level
    # This should be done selectively to avoid conflicts with API routes
    static_files = ['favicon.ico', 'manifest.json', 'robots.txt']
    for file in static_files:
        file_path = build_dir / file
        if file_path.exists():
            @app.get(f"/{file}")
            async def serve_static_file(file=file):
                return FileResponse(str(build_dir / file))
else:
    logger.error(f"Build directory not found at: {build_dir}")
    logger.error("Please run 'npm run build' or 'yarn build' to create the production build")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize services
manager = WebSocketManager()
pdf_service = PDFService({"pdf_output_dir": "pdfs"})

# Job status tracking
job_status = defaultdict(lambda: {
    "status": "pending",
    "result": None,
    "error": None,
    "debug_info": [],
    "company": None,
    "report": None,
    "last_update": datetime.now().isoformat()
})

# MongoDB initialization
mongodb = None
if mongo_uri := os.getenv("MONGODB_URI"):
    try:
        mongodb = MongoDBService(mongo_uri)
        logger.info("MongoDB integration enabled")
    except Exception as e:
        logger.warning(f"Failed to initialize MongoDB: {e}. Continuing without persistence.")

# Pydantic Models
class ResearchRequest(BaseModel):
    company: str
    company_url: str | None = None
    industry: str | None = None
    hq_location: str | None = None

class PDFGenerationRequest(BaseModel):
    report_content: str
    company_name: str | None = None

# API Routes
@app.get("/")
async def ping():
    """Health check endpoint"""
    return {"message": "Alive", "static_files_configured": build_dir.exists()}

@app.options("/research")
async def preflight():
    """Handle preflight requests for CORS"""
    response = JSONResponse(content=None, status_code=200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.post("/research")
async def research(data: ResearchRequest):
    """Initiate a new research job"""
    try:
        logger.info(f"Received research request for {data.company}")
        job_id = str(uuid.uuid4())
        asyncio.create_task(process_research(job_id, data))

        response = JSONResponse(content={
            "status": "accepted",
            "job_id": job_id,
            "message": "Research started. Connect to WebSocket for updates.",
            "websocket_url": f"/research/ws/{job_id}"
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    except Exception as e:
        logger.error(f"Error initiating research: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_research(job_id: str, data: ResearchRequest):
    """Process research job asynchronously"""
    try:
        if mongodb:
            mongodb.create_job(job_id, data.dict())
        
        await asyncio.sleep(1)  # Allow WebSocket connection
        await manager.send_status_update(job_id, status="processing", message="Starting research")

        graph = Graph(
            company=data.company,
            url=data.company_url,
            industry=data.industry,
            hq_location=data.hq_location,
            websocket_manager=manager,
            job_id=job_id
        )

        state = {}
        async for s in graph.run(thread={}):
            state.update(s)
        
        # Look for the compiled report in either location
        report_content = state.get('report') or (state.get('editor') or {}).get('report')
        
        if report_content:
            logger.info(f"Found report in final state (length: {len(report_content)})")
            job_status[job_id].update({
                "status": "completed",
                "report": report_content,
                "company": data.company,
                "last_update": datetime.now().isoformat()
            })
            
            if mongodb:
                mongodb.update_job(job_id=job_id, status="completed")
                mongodb.store_report(job_id=job_id, report_data={"report": report_content})
            
            await manager.send_status_update(
                job_id=job_id,
                status="completed",
                message="Research completed successfully",
                result={
                    "report": report_content,
                    "company": data.company
                }
            )
        else:
            logger.error(f"Research completed without finding report. State keys: {list(state.keys())}")
            logger.error(f"Editor state: {state.get('editor', {})}")
            
            error_message = state.get('error', 'No report found')
            
            await manager.send_status_update(
                job_id=job_id,
                status="failed",
                message="Research completed but no report was generated",
                error=error_message
            )

    except Exception as e:
        logger.error(f"Research failed: {str(e)}", exc_info=True)
        await manager.send_status_update(
            job_id=job_id,
            status="failed",
            message=f"Research failed: {str(e)}",
            error=str(e)
        )
        if mongodb:
            mongodb.update_job(job_id=job_id, status="failed", error=str(e))

@app.get("/research/pdf/{filename}")
async def get_pdf(filename: str):
    """Serve PDF files"""
    pdf_path = os.path.join("pdfs", filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type='application/pdf', filename=filename)

@app.websocket("/research/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time updates"""
    try:
        await websocket.accept()
        await manager.connect(websocket, job_id)

        if job_id in job_status:
            status = job_status[job_id]
            await manager.send_status_update(
                job_id,
                status=status["status"],
                message="Connected to status stream",
                error=status["error"],
                result=status["result"]
            )

        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect(websocket, job_id)
                break

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}", exc_info=True)
        manager.disconnect(websocket, job_id)

@app.get("/research/{job_id}")
async def get_research(job_id: str):
    """Get research job status"""
    if not mongodb:
        if job_id in job_status:
            return job_status[job_id]
        raise HTTPException(status_code=404, detail="Research job not found")
    
    job = mongodb.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Research job not found")
    return job

@app.get("/research/{job_id}/report")
async def get_research_report(job_id: str):
    """Get research report for a specific job"""
    if not mongodb:
        if job_id in job_status:
            result = job_status[job_id]
            if report := result.get("report"):
                return {"report": report}
        raise HTTPException(status_code=404, detail="Report not found")
    
    report = mongodb.get_report(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Research report not found")
    return report

@app.post("/generate-pdf")
async def generate_pdf(data: PDFGenerationRequest):
    """Generate a PDF from markdown content and stream it to the client"""
    try:
        success, result = pdf_service.generate_pdf_stream(data.report_content, data.company_name)
        if success:
            pdf_buffer, filename = result
            return StreamingResponse(
                pdf_buffer,
                media_type='application/pdf',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"'
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result)
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# React App Routes - MUST BE AFTER API ROUTES
@app.get("/platform")
@app.get("/platform/")
async def serve_platform_root():
    """Serve React app for /platform root"""
    index_path = build_dir / "index.html"
    
    if not index_path.exists():
        logger.error(f"index.html not found at: {index_path}")
        raise HTTPException(
            status_code=404, 
            detail=f"UI build not found. Please run 'npm run build' or 'yarn build'"
        )
    
    logger.debug(f"Serving React app from: {index_path}")
    return FileResponse(str(index_path), media_type="text/html")

@app.get("/platform/{full_path:path}")
async def serve_platform_spa(full_path: str):
    """Serve React app for all /platform/* routes (SPA support)"""
    # First, check if this is a request for a static file
    static_file_path = build_dir / full_path
    
    # If it's a file that exists (CSS, JS, images, etc.), serve it
    if static_file_path.exists() and static_file_path.is_file():
        logger.debug(f"Serving static file: {static_file_path}")
        # Determine media type based on file extension
        file_ext = static_file_path.suffix.lower()
        media_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon',
            '.woff': 'font/woff',
            '.woff2': 'font/woff2',
            '.ttf': 'font/ttf',
            '.eot': 'font/eot'
        }
        media_type = media_types.get(file_ext, 'application/octet-stream')
        return FileResponse(str(static_file_path), media_type=media_type)
    
    # Otherwise, serve index.html for client-side routing
    index_path = build_dir / "index.html"
    
    if not index_path.exists():
        logger.error(f"index.html not found at: {index_path}")
        raise HTTPException(
            status_code=404, 
            detail=f"UI build not found. Please run 'npm run build' or 'yarn build'"
        )
    
    logger.debug(f"Serving React app (SPA route) for path: /platform/{full_path}")
    return FileResponse(str(index_path), media_type="text/html")

# Optional: Debug route to check what's in the build directory
@app.get("/api/debug/static-files")
async def debug_static_files():
    """Debug endpoint to check static files configuration"""
    result = {
        "build_dir": str(build_dir),
        "build_dir_exists": build_dir.exists(),
        "build_dir_contents": [],
        "assets_dir_contents": []
    }
    
    if build_dir.exists():
        result["build_dir_contents"] = [str(f.name) for f in build_dir.iterdir()]
        
        assets_dir = build_dir / "assets"
        if assets_dir.exists():
            result["assets_dir_contents"] = [str(f.name) for f in assets_dir.iterdir()]
    
    return result

if __name__ == "__main__":
    # Additional startup logging
    logger.info("=" * 50)
    logger.info("Starting Tavily Company Research API")
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Script directory: {Path(__file__).parent}")
    logger.info(f"Build directory: {build_dir}")
    logger.info(f"Build directory exists: {build_dir.exists()}")
    
    if not build_dir.exists():
        logger.warning("!" * 50)
        logger.warning("BUILD DIRECTORY NOT FOUND!")
        logger.warning("Please run 'npm run build' or 'yarn build' to create the production build")
        logger.warning("!" * 50)
    
    logger.info("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")