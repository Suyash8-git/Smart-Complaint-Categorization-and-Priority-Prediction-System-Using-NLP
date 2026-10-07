"""Builds data/complaints.csv (7 categories x 100) with RULE-BASED priority labels.

Priority rule:  final = base severity of the issue
                        +1 if an urgency clause is present (exam today, emergency...)
                        -1 if a 'minor' clause is present (not urgent, sometimes...)
Levels: 0 Low, 1 Medium, 2 High, 3 Critical (clamped to 0..3).
Replace/extend this with real or hand-written complaints for a stronger project.
"""
import random
import pandas as pd

random.seed(42)
LEVELS = ["Low", "Medium", "High", "Critical"]

ISSUES = {
 "Internet/Wi-Fi": [("the wifi is not working",2),("the internet connection is completely down",2),("the network keeps disconnecting",1),("the internet speed is very slow",0),("unable to connect to the college wifi",1),("the router is not responding",2),("the wifi signal is very weak",0),("cannot access any website",1),("the lan cable is not connecting to the network",1),("the internet is lagging a lot",0),("the wifi password is not working",1),("there is no network connectivity",2)],
 "Hardware": [("the computer monitor is broken",1),("the keyboard keys are not working",1),("the mouse is not responding",0),("the projector is not turning on",1),("the printer is jammed",1),("the laptop battery is draining very fast",0),("the computer is not starting",2),("the cpu is making a loud noise",0),("the hard disk has crashed",2),("the screen is flickering",1),("the scanner is not working",0),("the server hardware has failed",3)],
 "Software": [("ms word is not opening",1),("the application keeps crashing",1),("python is not installed on the lab computers",1),("unable to login to the student portal",2),("the antivirus is showing a virus warning",2),("the software license has expired",1),("the operating system is very slow",0),("the browser is freezing",0),("unable to install the required software",1),("the exam software is not loading",2),("the database application shows an error",1),("the excel file is corrupted and will not open",1)],
 "Electricity": [("there is a power failure",2),("the lights are not working",1),("the power sockets are not working",1),("the tube light is flickering",0),("there is no electricity",2),("the ups is beeping continuously",1),("voltage fluctuation is damaging the devices",2),("the switch board is sparking",3),("the power supply keeps tripping",2),("the generator is not starting",1),("an electric wire is hanging loose",3),("the bulb has fused",0)],
 "Maintenance": [("the fan is making noise",0),("the air conditioner is not cooling",1),("the water cooler is leaking",1),("the door lock is broken",1),("the window glass is broken",1),("the ceiling is leaking water",2),("the bench is damaged",0),("the washroom is not clean",1),("the tap is leaking",0),("the whiteboard is damaged",0),("the drainage is blocked",2),("the room needs cleaning",0)],
 "Security": [("someone entered the restricted area without permission",3),("i lost my id card",1),("an unknown person is roaming around",2),("the cctv camera is not working",2),("my laptop was stolen",3),("someone tried to access my account without permission",2),("the entrance gate lock is broken",2),("there is a suspicious bag lying unattended",3),("there was unauthorized access to the server room",3),("the biometric attendance is not working",1),("a student is misusing another student's id card",1),("the security guard was not at the gate",1)],
 "Other": [("i need help with my id card",0),("the notice board has outdated information",0),("the canteen food quality is poor",1),("the library is closing at wrong timings",0),("the college bus is always late",1),("i want to change my timetable slot",0),("my fees receipt has not been issued",1),("the parking space is not sufficient",0),("my scholarship amount has not arrived",1),("i did not receive my exam hall ticket",2),("the faculty is not responding to my emails",0),("my certificate has not been issued",1)],
}
PLACES = ["", "", " in lab 3", " in classroom 204", " in the library", " in the hostel", " in the computer laboratory", " in the seminar hall", " in the department office", " in the main building", " in lab 1", " near the canteen", " in the examination hall", " in the server room"]
PREFIX = ["", "", "", "sir, ", "please note that ", "i want to report that ", "hello, ", "kindly check, "]
SUFFIX = ["", "", " since morning", " since yesterday", " for the last two days", " again today", " this week"]
URGENT = [" and our practical exam is today", " and the exam is about to start", " and this is an emergency", " and it needs immediate attention", " and students are unable to work", ", please fix it urgently", " and the placement test is starting soon"]
MINOR = [" but it is not urgent", " only sometimes", " but we can manage for now", " it is a minor problem"]


def build(n_per_class=100):
    rows = []
    for cat, issues in ISSUES.items():
        seen = set()
        while len(seen) < n_per_class:
            issue, base = random.choice(issues)
            r = random.random()
            mod = random.choice(URGENT) if r < 0.30 else random.choice(MINOR) if r < 0.45 else ""
            lvl = base + (1 if mod in URGENT else -1 if mod in MINOR else 0)
            lvl = max(0, min(3, lvl))
            text = (random.choice(PREFIX) + issue + random.choice(PLACES) + random.choice(SUFFIX) + mod).strip()
            text = text[0].upper() + text[1:]
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            rows.append((text, cat, LEVELS[lvl]))
    df = pd.DataFrame(rows, columns=["complaint", "category", "priority"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    df.to_csv("data/complaints.csv", index=False)
    print(df.shape); print(df.category.value_counts()); print(df.priority.value_counts()); print(df.head(8))
