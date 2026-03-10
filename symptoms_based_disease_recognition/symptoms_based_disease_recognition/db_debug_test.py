import os
import sys

BASE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE, "src"))

from database import db

print("DB config:", db.host, db.user, db.database)

print("Existing diseases before insert:")
print(db.get_all_diseases())

ok = db.add_disease("TestDiseaseCursor", "Test description", 3.0, 5)
print("add_disease returned:", ok)

print("Diseases after insert:")
print(db.get_all_diseases())

