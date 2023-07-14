import os

from dotenv import load_dotenv

from neat_ai_assistant.agent.tools import WeatherRetrievalTool

load_dotenv()
openweathermap_api_key = os.getenv("WEATHER_API_KEY")
weather_tool = WeatherRetrievalTool(openweathermap_api_key)
res = weather_tool.run(
    json_query={
        "location": "mönchengladbach",
        "datetime": "2023-07-17 10:45:12"})
print(res.results[0])
print("")
