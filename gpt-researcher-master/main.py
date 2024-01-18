from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from fastapi import Depends

from agent.llm_utils import choose_agent
from agent.run import WebSocketManager

# Define the database connection string
DATABASE_URL = "sqlite:///./test.db"  # Update with your actual database connection string

# Define the SQLAlchemy models
Base = declarative_base()

class ResearchResult(Base):
    __tablename__ = 'research_results'
    id = Column(Integer, primary_key=True, index=True)
    task = Column(String, index=True)
    report_type = Column(String, index=True)
    agent = Column(String, index=True)
    agent_role_prompt = Column(String, nullable=True)
    logs = Column(String)

# Function to create a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()
app.mount("/site", StaticFiles(directory="client"), name="site")
app.mount("/static", StaticFiles(directory="client/static"), name="static")

# Dynamic directory for outputs once the first research is run
@app.on_event("startup")
def startup_event():
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Configure the database session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Configure the templates
templates = Jinja2Templates(directory="client")

# Initialize WebSocketManager
manager = WebSocketManager()

# Modify the choose_agent function to use Hugging Face Llama2
def choose_agent(task):
    # Define the Hugging Face Llama2 API endpoint
    llama2_url = "https://llama2.huggingface.co/search"

    # Your Hugging Face Llama2 API key
    llama2_api_key = "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"

    # Your query to find the appropriate agent based on the task
    query = f"Find me an agent for {task}"

    # Set up the headers with your API key
    headers = {"Authorization": f"Bearer {llama2_api_key}"}

    # Make the request to Hugging Face Llama2
    response = requests.get(llama2_url, headers=headers, params={"query": query})

    if response.status_code == 200:
        # Process the response and extract the agent information
        agent_data = response.json()
        agent = agent_data.get("agent")
        agent_role_prompt = agent_data.get("agent_role_prompt")

        return {"agent": agent, "agent_role_prompt": agent_role_prompt}
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return {"agent": "Default Agent", "agent_role_prompt": None}

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request, "report": None})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("start"):
                json_data = json.loads(data[6:])
                task = json_data.get("task")
                report_type = json_data.get("report_type")
                agent = json_data.get("agent")
                
                # Temporary solution for "Auto Agent"
                if agent == "Auto Agent":
                    agent_dict = choose_agent(task)
                    agent = agent_dict.get("agent")
                    agent_role_prompt = agent_dict.get("agent_role_prompt")
                else:
                    agent_role_prompt = None

                await websocket.send_json({"type": "logs", "output": f"Initiated an Agent: {agent}"})
                
                if task and report_type and agent:
                    # Store the results in the database
                    result = ResearchResult(task=task, report_type=report_type, agent=agent, agent_role_prompt=agent_role_prompt, logs=f"Initiated an Agent: {agent}")
                    db.add(result)
                    db.commit()

                    await manager.start_streaming(task, report_type, agent, agent_role_prompt, websocket)
                else:
                    print("Error: not enough parameters provided.")

    except WebSocketDisconnect:
        await manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
``
