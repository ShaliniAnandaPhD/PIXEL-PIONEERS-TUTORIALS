# Python Growing Food on Mars: Automated Seed Planting and Germination Tracking

# Welcome to the twelfth class in the "Growing Food on Mars" series!
# In this class, we will learn how to develop a Python script for automated seed planting and germination tracking.

# First, let's create a class to represent a seed:
class Seed:
   def __init__(self, name, days_to_germinate):
       self.name = name
       self.days_to_germinate = days_to_germinate
       self.planted = False
       self.germinated = False

   def plant(self, day):
       self.planted = True
       self.planting_day = day
       print(f"{self.name} seed planted on day {day}.")

   def check_germination(self, day):
       if self.planted and not self.germinated:
           if day - self.planting_day >= self.days_to_germinate:
               self.germinated = True
               print(f"{self.name} seed germinated on day {day}.")
       return self.germinated

# Let's create a function to automate seed planting:
def automate_planting(seeds, planting_schedule):
   for day, seed_name in planting_schedule.items():
       for seed in seeds:
           if seed.name == seed_name:
               seed.plant(day)
               break

# Let's create a function to track germination:
def track_germination(seeds, num_days):
   for day in range(1, num_days + 1):
       print(f"\nDay {day}:")
       for seed in seeds:
           seed.check_germination(day)

# Let's create some sample seeds:
tomato_seed = Seed("Tomato", 7)
lettuce_seed = Seed("Lettuce", 4)
carrot_seed = Seed("Carrot", 10)

seeds = [tomato_seed, lettuce_seed, carrot_seed]

# Let's define a planting schedule:
planting_schedule = {
   1: "Tomato",
   3: "Lettuce",
   5: "Carrot"
}

# Let's automate seed planting based on the schedule:
print("Automating seed planting...")
automate_planting(seeds, planting_schedule)

# Let's track germination for 15 days:
print("\nTracking germination for 15 days...")
track_germination(seeds, 15)

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different seed types, germination times, and planting schedules to further explore automated seed planting and germination tracking.

# In the next class, we will explore how to create a Python-based system for pollination in a Martian greenhouse.
# Get ready to dive deeper into the world of plant reproduction on Mars!
