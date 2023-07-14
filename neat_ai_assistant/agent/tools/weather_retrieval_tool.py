from pydantic import BaseModel
import requests

from typing import List, Optional, Tuple
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.location import Location

from ..tool import Tool, ToolParam, ToolResult


tool_param_location = ToolParam(
    name="location",
    type="string",
    description="The location to get the weather for.",
    required=True
)
datetime_format = "%Y-%m-%d %H:%M:%S"
current_datetime = datetime.now().strftime(datetime_format)
tool_param_datetime = ToolParam(
    name="datetime",
    type="string",
    description=f"Current datetime: \"{current_datetime}\". Return the desired time associated with the weather request in this format: \"{datetime_format}\".",
    required=True
)


class WeatherResult(BaseModel):
    city: str
    country: str
    sunrise_today: datetime
    sunset_today: datetime
    forecast_datetime: datetime
    temperature_celsius: float
    humidity_pct: int
    weather_description: str
    wind_speed_km_h: float
    wind_direction: str

    def to_string(self):
        return f"""1. Location Information
Location: {self.city} ({self.country})
Sunrise (today): {self.sunrise_today.strftime(datetime_format)}
Sunset (today): {self.sunset_today.strftime(datetime_format)}

2. Forecast Information
Date and time: {self.forecast_datetime.strftime(datetime_format)}
Temperature: {str(self.temperature_celsius)}°C
Humidity: {str(self.humidity_pct)}%
Weather Description: {self.weather_description}
Wind speed: {self.wind_speed_km_h} km/h
Wind direction: {self.wind_direction}"""


class WeatherRetrievalTool(Tool):
    def __init__(
        self,
        open_weather_map_api_key: str,
        name: str = "Weather Retrieval API",
        description: str = "Retrieve the current weather for a location.",
        params: List[ToolParam] = [
            tool_param_location,
            tool_param_datetime
        ]
    ):
        super().__init__(name, description, params)
        self.api_key = open_weather_map_api_key

    @staticmethod
    def _get_coordinates(location_name: str) -> Optional[Tuple[float, float]]:
        geolocator = Nominatim(user_agent="my-app")
        location: Location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        else:
            return None

    @staticmethod
    def _build_weather_result(
            weather_info: dict,
            city_info: dict) -> WeatherResult:
        def degrees_to_cardinal(d: int):
            dirs = [
                'N',
                'NNE',
                'NE',
                'ENE',
                'E',
                'ESE',
                'SE',
                'SSE',
                'S',
                'SSW',
                'SW',
                'WSW',
                'W',
                'WNW',
                'NW',
                'NNW']
            ix = round(d / (360. / len(dirs)))
            return dirs[ix % len(dirs)]

        return WeatherResult(
            city=city_info["name"],
            country=city_info["country"],
            sunrise_today=datetime.fromtimestamp(city_info["sunrise"]),
            sunset_today=datetime.fromtimestamp(city_info["sunset"]),
            forecast_datetime=datetime.strptime(
                weather_info['dt_txt'], '%Y-%m-%d %H:%M:%S'),
            temperature_celsius=round(
                weather_info["main"]["temp"] -
                273.15,
                1),
            # Kelvin to Celsius conversion
            humidity_pct=weather_info["main"]["humidity"],
            weather_description=weather_info["weather"][0]["description"],
            wind_speed_km_h=round(
                weather_info["wind"]["speed"] * 3.6,
                1),
            # m/s to km/h conversion
            wind_direction=degrees_to_cardinal(weather_info["wind"]["deg"])
        )

    def _get_weather(
            self,
            lat: float,
            lon: float,
            desired_datetime: str) -> WeatherResult:
        url = "https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"
        request_url = url.format(
            lat=str(lat),
            lon=str(lon),
            api_key=self.api_key
        )
        response = requests.get(request_url)
        response_json = response.json()

        def to_datetime(datetime_str):
            try:
                return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None

        desired_datetime = to_datetime(desired_datetime) or current_datetime
        weather_info, city_info = response_json["list"], response_json["city"]
        relevant_weather_info = min(
            weather_info,
            key=lambda d: abs(
                desired_datetime -
                datetime.strptime(
                    d['dt_txt'],
                    '%Y-%m-%d %H:%M:%S')))
        return self._build_weather_result(relevant_weather_info, city_info)

    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)

        coordinates = self._get_coordinates(json_query["location"])
        if not bool(coordinates):
            results = []
        else:
            lat, lon = coordinates
            weather_result = self._get_weather(
                lat, lon, json_query["datetime"])
            results = [weather_result.to_string()]

        return self.build_tool_result(results)
