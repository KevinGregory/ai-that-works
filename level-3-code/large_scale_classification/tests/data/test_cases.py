"""Test cases for the large-scale classification system.

This file contains comprehensive test cases covering different categories
from the current category set. These test cases include realistic product
descriptions and expected category classifications.
"""

from typing import TypedDict


class TestCase(TypedDict):
    """Test case."""

    text: str
    category: str
    test_type: str


tests: list[TestCase] = [
    {
        "text": "Samsung Counter-Depth 17.5-cu ft 3-Door Smart Compatible French Door Refrigerator with Ice Maker (Fingerprint Resistant Matte Black Steel) ENERGY STAR Certified",
        "category": "/Appliances/Refrigerators/French Door Refrigerators",
        "test_type": "llm_generated",
    },
    {
        "text": 'Bosch 800 Series 24" Stainless Steel Built-In Dishwasher with Third Rack and CrystalDry Technology',
        "category": "/Appliances/Dishwashers/Built-In Dishwashers",
        "test_type": "llm_generated",
    },
    {
        "text": "BLACK+DECKER 6-Place Setting Compact Countertop Dishwasher in White",
        "category": "/Appliances/Dishwashers/Countertop Dishwashers",
        "test_type": "llm_generated",
    },
    {
        "text": "GE Portable Dishwasher with Stainless Steel Interior and Wheels",
        "category": "/Appliances/Dishwashers/Portable Dishwashers",
        "test_type": "llm_generated",
    },
    {
        "text": "Hobart LXER-2 Undercounter Commercial Dishwasher with Built-in Booster Heater",
        "category": "/Appliances/Dishwashers/Commercial Dishwashers",
        "test_type": "llm_generated",
    },
    {
        "text": "InSinkErator Evolution Compact 3/4 HP Garbage Disposal with SoundSeal Technology",
        "category": "/Appliances/Garbage Disposals",
        "test_type": "llm_generated",
    },
    {
        "text": "Whirlpool Dishwasher Upper Dish Rack Assembly W10350375",
        "category": "/Appliances/Appliance Parts/Dishwasher Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "KitchenAid Stand Mixer Bowl Lift Lever and Spring Assembly",
        "category": "/Appliances/Appliance Parts/Small Appliance Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Samsung DA29-00020B HAF-CIN/EXP Refrigerator Water Filter",
        "category": "/Appliances/Appliance Parts/Refrigerator Water Filters",
        "test_type": "llm_generated",
    },
    {
        "text": "LG ADQ36006101 Refrigerator Air Filter for French Door Models",
        "category": "/Appliances/Appliance Parts/Refrigerator Air Filters",
        "test_type": "llm_generated",
    },
    {
        "text": "GE WR17X12633 Refrigerator Ice Maker Assembly",
        "category": "/Appliances/Appliance Parts/Refrigerator Parts",
        "test_type": "llm_generated",
    },
    {
        "text": 'Broan-NuTone Range Hood Grease Filter 11-3/4" x 14-1/4" Aluminum',
        "category": "/Appliances/Appliance Parts/Range Hood Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Frigidaire 316075103 Oven Bake Element 2500 Watts",
        "category": "/Appliances/Appliance Parts/Oven Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "InSinkErator Garbage Disposal Splash Guard and Stopper",
        "category": "/Appliances/Appliance Parts/Garbage Disposal Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Sharp Carousel Microwave Glass Turntable Plate 12.5 Inch",
        "category": "/Appliances/Appliance Parts/Microwave Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Whirlpool W10715708 Ice Maker Kit for Top-Freezer Refrigerators",
        "category": "/Appliances/Appliance Parts/Ice Maker Kits",
        "test_type": "llm_generated",
    },
    {
        "text": "GE WB31T10013 Cooktop Burner Drip Pan Set Chrome",
        "category": "/Appliances/Appliance Parts/Cooktop Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Whirlpool W10116794 Stove Burner Control Knob Black",
        "category": "/Appliances/Appliance Parts/Stove Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Frigidaire 5304505209 Freezer Door Gasket Seal",
        "category": "/Appliances/Appliance Parts/Freezer Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "NewAir Wine Cooler Replacement Shelves AWR-460DB Set of 6",
        "category": "/Appliances/Appliance Parts/Wine Cooler Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Whirlpool W10837240 Dryer Lint Screen Filter",
        "category": "/Appliances/Appliance Parts/Dryer Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "GE WC22X10047 Trash Compactor Bags 15-Pack",
        "category": "/Appliances/Appliance Parts/Trash Compactor Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Frigidaire 5304505209 Dehumidifier Water Collection Bucket",
        "category": "/Appliances/Appliance Parts/Dehumidifier Parts",
        "test_type": "llm_generated",
    },
    {
        "text": "Samsung SKK-DD Washer Dryer Stacking Kit with Pull-Out Shelf",
        "category": "/Appliances/Appliance Parts/Washer and Dryer Stacking Kits",
        "test_type": "llm_generated",
    },
    {
        "text": "Shark Navigator Vacuum Belt 2-Pack XB2950",
        "category": "/Appliances/Appliance Parts/Vacuum Parts/Vacuum Belts",
        "test_type": "llm_generated",
    },
    {
        "text": "Small heating/cooling unit",
        "category": "/Heating, Venting & Cooling/Mini Split Air Conditioners/Mini Split ACs",
        "test_type": "human_generated",
    },
    {
        "text": "latex gloves",
        "category": "/Safety Equipment/Disposable Protective Clothing/Disposable Gloves",
        "test_type": "human_generated",
    },    
    {
        "text": "flourescent bulbs",
        "category": "/Lighting/Light Bulbs/CFL Bulbs",
        "test_type": "human_generated",
    },
    {
        "text": "wall lamp",
        "category": "/Lighting/Wall Sconces",
        "test_type": "human_generated",
    },
    {
        "text":"over door shoe rack",
        "category": "/Storage & Organization/Shoe Storage/Hanging Shoe Organizers",
        "test_type": "human_generated",
    },
    {
        "text": "ping-pong table",
        "category": "/Sports & Outdoors/Games/Game Room/Ping Pong Tables",
        "test_type": "human_generated",
    },
    {
        "text": "eye bolt",
        "category": "/Hardware/Fasteners/Bolts/Eye Bolts",
        "test_type": "human_generated",
    },
    {
        "text": "cloth to use under painting to prevent mess",
        "category": "/Paint/Paint Supplies/Drop Cloths",
        "test_type": "human_generated",
    },
    {
        "text": "power equipment",
        "category": "/Outdoors/Outdoor Power Equipment",
        "test_type": "human_generated",
    },
    {
        "text": "desk shelves",
        "category": "/Storage & Organization/Office Storage & Organization",
        "test_type": "human_generated",
    },
    {
        "text": "paddleboard",
        "category": "/Sports & Outdoors/Boating/Water Sports/Stand Up Paddleboards",
        "test_type": "human_generated",
    },
    {
        "text": "backyard golf course",
        "category": "/Sports & Outdoors/Outdoor Sports/Golf Equipment/Putting Greens",
        "test_type": "human_generated",
    },
    {
        "text": "Stove with red knobs",
        "category": "/Appliances/Ranges/Gas Ranges/Double Oven Gas Ranges",
        "test_type": "human_generated",
    },
    {
        "text": "Refrigerator with hidden door with built in ice",
        "category": "/Appliances/Refrigerators/French Door Refrigerators",
        "test_type": "human_generated",
    },
    {
        "text": "nest thermostat",
        "category": "/Smart Home/Smart Devices/Smart Thermostats",
        "test_type": "human_generated",
    },
    {
        "text": "Silver titanium top load washing machine",
        "category": "/Appliances/Washers & Dryers/Washing Machines",
        "test_type": "human_generated",
    },
    {
        "text": "Smeg toaster",
        "category": "/Appliances/Small Kitchen Appliances/Toasters",
        "test_type": "human_generated",
    },
    {
        "text": "fire protection document safe",
        "category": "/Tools/Safety & Security/Safes/Home Safes",
        "test_type": "human_generated",
    },
    {
        "text": "walkie talkie",
        "category": "/Electrical/Electronics/Two-Way Radios",
        "test_type": "human_generated",
    },
    {
        "text": "backyard shed",
        "category": "/Storage & Organization/Outdoor Storage/Sheds",
        "test_type": "human_generated",
    },
    {
        "text": "suspenders",
        "category": "/Workwear/Workwear Accessories/Work Suspenders",
        "test_type": "human_generated",
    },
    {
        "text": "masking tape",
        "category": "/Paint/Paint Supplies/Tape/Masking Tape",
        "test_type": "human_generated",
    },
    {
        "text": "backyard fireplace",
        "category": "/Outdoors/Outdoor Heating/Outdoor Fireplaces",
        "test_type": "human_generated",
    },
    {
        "text": "carbon pre-filter",
        "category": "/Heating, Venting & Cooling/Air Purifiers",
        "test_type": "human_generated",
    },
    {
        "text": "wire",
        "category": "/Electrical/Wire",
        "test_type": "human_generated",
    },    
    {
        "text":"Car battery",
        "category": "/Automotive/Battery Charging Systems/Car Batteries",
        "test_type": "human_generated",
    },    
    {
        "text":"radiator fluid",
        "category": "/Automotive/Car Fluids & Chemicals",
        "test_type": "human_generated",
    },    
    {
        "text":"Auto Light Bulb",
        "category": "/Automotive/Auto Parts/Car Lights",
        "test_type": "human_generated",
    },    
    {
        "text":"Step Ladder",
        "category": "/Building Materials/Ladders/Step Ladders",
        "test_type": "human_generated",
    },    
    {
        "text":"Toilet Flapper Valve",
        "category": "/Plumbing/Plumbing Parts/Toilet Parts/Toilet Repair Kits",
        "test_type": "human_generated",
    },    
    {
        "text":"Light Bulbs",
        "category": "/Lighting/Light Bulbs",
        "test_type": "human_generated",
    },    
    {
        "text":"light switch",
        "category": "/Electrical/Wall Plates/Light Switch Plates",
        "test_type": "human_generated",
    },    
    {
        "text":"bathroom decoration",
        "category": "/Bath/Bathroom Accessories/Bathroom Decor",
        "test_type": "human_generated",
    },    
    {
        "text":"space heater",
        "category": "/Heating, Venting & Cooling/Heaters/Space Heaters",
        "test_type": "human_generated",
    },    
    {
        "text":"welding mask/helmet",
        "category": "/Tools/Welding & Soldering/Welding Safety Apparel/Welding Helmets",
        "test_type": "human_generated",
    },    
    {
        "text":"natural gas detector",
        "category": "/Electrical/Fire Safety/Fire Safety Accessories",
        "test_type": "human_generated",
    },
    {
        "text":"ice maker",
        "category": "/Appliances/Appliance Parts/Ice Maker Kits",
        "test_type": "human_generated",
    },
    {
        "text":"microwave in drawer",
        "category": "/Appliances/Microwaves",
        "test_type": "human_generated",
    },
    {
        "text":"induction stove",
        "category": "/Appliances/Cooktops/Induction Cooktops",
        "test_type": "human_generated",
    },
    {
        "text":"Front loading washing machine",
        "category": "/Appliances/Washers & Dryers/Washing Machines",
        "test_type": "human_generated",
    },
    {
        "text":"Toaster oven with airfry",
        "category": "/Appliances/Small Kitchen Appliances/Toasters",
        "test_type": "human_generated",
    },
    {
        "text":"rice cooker",
        "category": "/Appliances/Small Kitchen Appliances/Cookers",
        "test_type": "human_generated",
    },
    {
        "text":"crockpot",
        "category": "/Appliances/Small Kitchen Appliances/Cookers",
        "test_type": "human_generated",
    }    
]
