from __future__ import annotations
from typing import Annotated

import logging

from livekit.agents import llm

logger = logging.getLogger(__name__)

class Actions(llm.FunctionContext):
    """Actions"""

    def __init__(self):
        super().__init__()

    @llm.ai_callable()
    async def get_flight_info(self,
        flight_number: Annotated[str, llm.TypeInfo(description="Airline flight number")]):
        """
        Get flight information for a given flight number

        Args:
            flight_number: Airline flight number

        Returns:
            Flight information
        """
        return {"flight_number": flight_number, "departure": "LAX", "arrival": "JFK", "date": "2024-01-01", "time": "10:00 AM", "status": "On time"}

    @llm.ai_callable()
    async def get_booking_info(self, booking_number: Annotated[str, llm.TypeInfo(description="Airline booking number")]):
        """
        Get booking information for a given booking number

        Args:
            booking_number: Airline booking number

        Returns:
            Booking information
        """
        return {"booking_number": booking_number, 
                "flight_number": "AE" + booking_number, 
                "passenger_name": "John Doe",
                "passenger_email": "john.doe@example.com",
                "passenger_phone": "1234567890",
                "passenger_address": "123 Main St, Anytown, USA",
                "passenger_dob": "1990-01-01",
                "passenger_gender": "Male"
                }

