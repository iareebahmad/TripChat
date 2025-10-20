import os
import json
import requests
from dotenv import load_dotenv
import streamlit as st
from twilio.rest import Client
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()


class TripState(dict):
    query: str
    destination: str
    days: int
    interests: list
    itinerary: str
    weather: dict
    user_phone: str

def get_weather(destination: str, days: int = 3):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or not destination:
        return {}
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": destination, "appid": api_key, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("cod") != "200":
            return {}
        forecasts = {}
        for item in data.get("list", []):
            date = item["dt_txt"].split(" ")[0]
            if date not in forecasts:
                forecasts[date] = {
                    "temp": item["main"]["temp"],
                    "desc": item["weather"][0]["description"]
                }
            if len(forecasts) >= days:
                break
        return forecasts
    except Exception as e:
        print("Weather API error:", e)
        return {}

def send_whatsapp_message(to_number: str, message: str):
    client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    max_len = 1500  # slightly below 1600 to be safe
    chunks = [message[i:i+max_len] for i in range(0, len(message), max_len)]
    for chunk in chunks:
        client.messages.create(
            from_="whatsapp:+14155238886",
            to=f"whatsapp:{to_number}",
            body=chunk
        )


def extract_details_node(state: TripState):
    llm = ChatOpenAI(temperature=0)
    prompt = f"""
    Extract destination, number of days, and interests from:
    "{state['query']}"
    Respond as JSON: {{ "destination": "", "days": 0, "interests": [] }}
    """
    resp = llm.predict(prompt)
    try:
        parsed = json.loads(resp.replace("'", '"'))
    except:
        parsed = {}
    state['destination'] = parsed.get("destination", "Unknown")
    state['days'] = parsed.get("days", 3)
    state['interests'] = parsed.get("interests", [])
    return state

def weather_node(state: TripState):
    state['weather'] = get_weather(state['destination'], state['days'])
    return state

def plan_itinerary_node(state: TripState):
    llm = ChatOpenAI(temperature=0.2)
    weather_text = "\n".join(
        [f"{d}: {w['desc']} ({w['temp']}°C)" for d, w in state['weather'].items()]
    )
    prompt = f"""
    You are TripChat — an AI travel planner.
    Destination: {state['destination']}
    Days: {state['days']}
    Interests: {', '.join(state['interests'])}
    Weather: {weather_text}
    Create a {state['days']}-day itinerary in a friendly tone.
    """
    state['itinerary'] = llm.predict(prompt)
    return state

def send_whatsapp_node(state: TripState):
    if state.get("user_phone"):
        send_whatsapp_message(state["user_phone"], state["itinerary"])
    return state


graph = StateGraph(TripState)
graph.add_node("extract", extract_details_node)
graph.add_node("weather", weather_node)
graph.add_node("plan", plan_itinerary_node)
graph.add_node("whatsapp", send_whatsapp_node)

graph.add_edge("extract", "weather")
graph.add_edge("weather", "plan")
graph.add_edge("plan", "whatsapp")
graph.add_edge("whatsapp", END)

graph.set_entry_point("extract")
trip_agent = graph.compile()


st.title("TripChat")
st.subheader("Your AI Travel Agent")

query = st.text_input("Tell me about your trip:")
phone = st.text_input("Your WhatsApp number (with country code: +YYXXXXXXXXXX):")

if st.button("Generate & Send"):
    if query and phone:
        init_state = TripState(query=query, user_phone=phone)
        with st.spinner("Planning your trip..."):
            result = trip_agent.invoke(init_state)
        st.success("✅ Itinerary created and sent to your WhatsApp!")
        st.write(result['itinerary'])
