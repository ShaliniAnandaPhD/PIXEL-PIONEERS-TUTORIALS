
# Use case: Nutrition - Healthy Recipe Generation

import openai
from transformers import GPT2Tokenizer

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY"

# Define the GPT-4 model and tokenizer
model_engine = "gpt-4"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the prompt for generating healthy recipes
prompt = """
Generate a healthy recipe with the following criteria:
- Low in calories and fat
- High in protein and fiber
- Includes vegetables and lean protein
- Easy to prepare

Recipe:
"""

# Generate healthy recipes using GPT-4
def generate_recipe(prompt, max_tokens=200, num_recipes=1):
    recipes = []
    for _ in range(num_recipes):
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        )
        recipe = response.choices[0].text.strip()
        recipes.append(recipe)
    return recipes

# Generate multiple healthy recipes
num_recipes = 3
generated_recipes = generate_recipe(prompt, max_tokens=300, num_recipes=num_recipes)

# Print the generated recipes
for i, recipe in enumerate(generated_recipes):
    print(f"Recipe {i+1}:")
    print(recipe)
    print("---")
