import os
import uuid
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

connection = MongoClient(os.getenv("MONGODB_URI"))
db = connection[os.getenv("MONGODB_DATABASE")]

agents_collection = db["agents"]


def create_agent(user_id, agent_name, agent_description, agent_action):
    if agents_collection.find_one({"user_id": user_id, "agent_name": agent_name}):
        return {"status": "error", "message": "Agent already exists"}

    agents_collection.insert_one({
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "agent_name": agent_name,
        "agent_description": agent_description,
        "agent_action": agent_action,
        "is_custom": True
    })
    return {"status": "success"}


def get_agents(user_id):
    agents = agents_collection.find({"user_id": user_id})
    return [agent for agent in agents]


def get_agent(user_id, id):
    agent = agents_collection.find_one(
        {"user_id": user_id, "id": id})
    return agent


def delete_agent(user_id, id):
    agents_collection.delete_one(
        {"user_id": user_id, "id": id})
    return {"status": "success"}


def update_agent(user_id, id, agent_name, agent_description, agent_action):
    agent = get_agent(user_id, id)

    if not agent:
        return {"status": "error", "message": "Agent not found"}

    if agent_description is None:
        agent_description = agent["agent_description"]

    if agent_action is None:
        agent_action = agent["agent_action"]

    if agent_name is None:
        agent_name = agent["agent_name"]

    agents_collection.update_one(
        {"user_id": user_id, "id": id},
        {"$set": {"agent_description": agent_description,
                  "agent_action": agent_action, "agent_name": agent_name}}
    )
    return {"status": "success"}
