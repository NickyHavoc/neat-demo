from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import requests
from pydantic import BaseModel

from ..tool import Tool, ToolParam, ToolResult

try:
    from geopy.geocoders import Nominatim  # type: ignore
    from geopy.location import Location  # type: ignore

    GEOPY_AVAILABLE = True
except:
    GEOPY_AVAILABLE = False

TOOL_PARAM_LOCATION = ToolParam(
    name="location",
    type="string",
    description="The location to get the weather for.",
    required=True,
)
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CURRENT_DATETIME = datetime.now()  # TODO: should be somewhere else
TOOL_PARAM_DATETIME = ToolParam(
    name="datetime",
    type="string",
    description=f'Current datetime: "{CURRENT_DATETIME.strftime(DATETIME_FORMAT)}". Return the desired time associated with the weather request in this format: "{DATETIME_FORMAT}".',
    required=True,
)


class _WeatherResult(BaseModel):
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

    def to_string(self) -> str:
        return f"""1. Location Information
Location: {self.city} ({self.country})
Sunrise (today): {self.sunrise_today.strftime(DATETIME_FORMAT)}
Sunset (today): {self.sunset_today.strftime(DATETIME_FORMAT)}

2. Forecast Information
Date and time: {self.forecast_datetime.strftime(DATETIME_FORMAT)}
Temperature: {str(self.temperature_celsius)}°C
Humidity: {str(self.humidity_pct)}%
Weather Description: {self.weather_description}
Wind speed: {self.wind_speed_km_h} km/h
Wind direction: {self.wind_direction}"""


class WeatherRetrievalTool(Tool):
    """
    Uses OpenWeatherMap to retrieve weather information.
    Params:
    - location (str): location of interest (will be transformed into coordinates)
    - datetime (str): date and time to retrieve info for
    """

    def __init__(
        self,
        open_weather_map_api_key: str,
        name: str = "Weather Retrieval API",
        description: str = "Retrieve the current weather for a location.",
        params: Sequence[ToolParam] = [TOOL_PARAM_LOCATION, TOOL_PARAM_DATETIME],
    ) -> None:
        super().__init__(name, description, params)
        self.api_key = open_weather_map_api_key

    @staticmethod
    def _get_coordinates(location_name: str) -> Optional[tuple[float, float]]:
        geolocator = Nominatim(user_agent="my-app")
        location: Location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        else:
            return None

    @staticmethod
    def _build_weather_result(
        weather_info: Mapping[str, Any], city_info: Mapping[str, Any]
    ) -> _WeatherResult:
        def degrees_to_cardinal(d: int) -> str:
            dirs = [
                "N",
                "NNE",
                "NE",
                "ENE",
                "E",
                "ESE",
                "SE",
                "SSE",
                "S",
                "SSW",
                "SW",
                "WSW",
                "W",
                "WNW",
                "NW",
                "NNW",
            ]
            ix = round(d / (360.0 / len(dirs)))
            return dirs[ix % len(dirs)]

        return _WeatherResult(
            city=city_info["name"],
            country=city_info["country"],
            sunrise_today=datetime.fromtimestamp(city_info["sunrise"]),
            sunset_today=datetime.fromtimestamp(city_info["sunset"]),
            forecast_datetime=datetime.strptime(
                weather_info["dt_txt"], "%Y-%m-%d %H:%M:%S"
            ),
            temperature_celsius=round(weather_info["main"]["temp"] - 273.15, 1),
            # Kelvin to Celsius conversion
            humidity_pct=weather_info["main"]["humidity"],
            weather_description=weather_info["weather"][0]["description"],
            wind_speed_km_h=round(weather_info["wind"]["speed"] * 3.6, 1),
            # m/s to km/h conversion
            wind_direction=degrees_to_cardinal(weather_info["wind"]["deg"]),
        )

    def _get_weather(
        self, lat: float, lon: float, desired_datetime_str: str
    ) -> _WeatherResult:
        url = "https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"
        request_url = url.format(lat=str(lat), lon=str(lon), api_key=self.api_key)
        response = requests.get(request_url)
        response_json = response.json()

        def to_datetime(datetime_str: str) -> datetime:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

        desired_datetime = to_datetime(desired_datetime_str) or CURRENT_DATETIME
        weather_info, city_info = response_json["list"], response_json["city"]
        relevant_weather_info = min(
            weather_info,
            key=lambda d: abs(
                desired_datetime - datetime.strptime(d["dt_txt"], "%Y-%m-%d %H:%M:%S")
            ),
        )
        return self._build_weather_result(relevant_weather_info, city_info)

    def _run(self, json_query: Mapping[str, Any]) -> ToolResult:
        if not GEOPY_AVAILABLE:
            raise RuntimeError(
                "This tool requires 'geopy'. Please install the extra 'tool-extension'."
            )

        coordinates = self._get_coordinates(json_query["location"])
        if coordinates is None:
            results = []
        else:
            lat, lon = coordinates
            weather_result = self._get_weather(lat, lon, json_query["datetime"])
            results = [weather_result.to_string()]

        return self.to_result(results)
