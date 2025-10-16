# Define a simple class called 'Animal'
class Animal:
    # The __init__ method is called when a new object is created
    def __init__(self, name, sound):
        self.name  = name   # Each animal has a name
        self.sound = sound  # Each animal makes a sound

    # A method to make the animal speak
    def speak(self):
        print(f"{self.name} says: {self.sound}!")


# Create two Animal objects
cat = Animal("Whiskers", "Meow")
butterfly = Animal("Flutter", "...")

# Call the speak method for each animal
cat.speak()
butterfly.speak()


# --- Inheritance Example ---
# Define a 'Dog' class that inherits from 'Animal'
class Dog(Animal):
    # The __init__ method for Dog
    def __init__(self, name):
        # Call the parent class's __init__ method using the "super" keyword
        super().__init__(name, "Woof")

    # Add a new method specific to Dog
    def fetch(self):
        print(f"{self.name} is fetching the ball!")

# Create a Dog object
dog = Dog("Buddy")
dog.speak() 
dog.fetch()