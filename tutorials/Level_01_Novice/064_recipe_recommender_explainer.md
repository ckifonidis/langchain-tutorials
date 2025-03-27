# Recipe Recommender with LangChain: Complete Guide

## Introduction

This guide explores the implementation of a recipe recommendation system using LangChain's chat models and structured output parsing. The system provides personalized recipe suggestions based on available ingredients and user preferences.

Real-World Value:
- Personalized recipe recommendations
- Detailed cooking instructions
- Dietary preference handling
- Time-based meal planning

## Core LangChain Concepts

### 1. Chat Models

Chat models enable natural recipe recommendations:

```python
self.chat_model = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    temperature=0.7  # Some creativity in recipes
)

self.prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert chef..."),
    ("human", "Available ingredients: {ingredients}...")
])
```

Key Features:
1. **Natural Interaction**: Chef-like recipe suggestions
2. **Contextual Understanding**: Considers ingredients and preferences
3. **Creative Variations**: Balanced with temperature setting
4. **Role-Based Responses**: Expert chef persona

### 2. Structured Output

Structured output ensures consistent recipe format:

```python
class Recipe(BaseModel):
    name: str = Field(description="Recipe name")
    cuisine_type: str = Field(description="Type of cuisine")
    difficulty: str = Field(description="Easy/Medium/Hard")
    steps: List[RecipeStep] = Field(description="Cooking instructions")
```

Implementation Benefits:
1. **Type Safety**: Strong typing for recipe data
2. **Data Validation**: Automatic field validation
3. **Clear Structure**: Well-defined recipe format
4. **Easy Processing**: Structured recipe handling

## Implementation Components

### 1. Recipe Structure

```python
class RecipeStep(BaseModel):
    step_number: int = Field(...)
    instruction: str = Field(...)
    duration_minutes: Optional[int] = Field(...)
    temperature_celsius: Optional[int] = Field(...)
```

Key Features:
1. **Step Organization**: Numbered instructions
2. **Time Management**: Duration tracking
3. **Temperature Control**: Cooking temperatures
4. **Optional Details**: Flexible step information

### 2. Recipe Generation

```python
def get_recipe(
    self,
    ingredients: List[str],
    preferences: List[str],
    max_time: int
) -> Recipe:
    chain = self.prompt | self.chat_model | self.parser
    recipe = chain.invoke(input_data)
```

Generation Features:
1. **Input Processing**: Ingredient and preference handling
2. **Chain Composition**: Clear processing flow
3. **Structured Output**: Validated recipe format
4. **Error Handling**: Robust error management

## Expected Output

When running the Recipe Recommender, you'll see:

```
Scenario 1: Quick Vegetarian Meal
---------------------------------
Ingredients: pasta, tomatoes, garlic, olive oil, basil, parmesan cheese
Preferences: vegetarian, quick, easy
Time Available: 30 minutes

Recommended Recipe:
Name: Quick Basil Pesto Pasta
Cuisine: Italian
Difficulty: Easy
Time: 25 minutes
Servings: 4

Instructions:
1. Boil water for pasta (2 mins)
2. Cook pasta al dente (10 mins)
3. Prepare garlic and basil (5 mins)
4. Make sauce and combine (8 mins)
```

For a different scenario:
```
Scenario 2: Gluten-Free Dinner
-----------------------------
Recommended Recipe:
Name: Asian Chicken Stir-Fry
Cuisine: Asian Fusion
Difficulty: Medium
Time: 40 minutes
```

## Best Practices

### 1. Recipe Generation
- Validate ingredients
- Consider cooking times
- Include dietary tags
- Add cooking tips

### 2. Implementation
- Use strong typing
- Handle missing data
- Validate temperatures
- Add error handling

### 3. Output Format
- Clear instructions
- Time estimates
- Temperature guides
- Helpful tips

## References

1. LangChain Documentation:
   - [Chat Models](https://python.langchain.com/docs/modules/model_io/models/chat)
   - [Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers)
   - [Pydantic Output](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic)

2. Implementation Resources:
   - [Type Validation](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)
   - [Chat Templates](https://python.langchain.com/docs/modules/model_io/prompts/chat)
   - [Response Formats](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic)

3. Additional Resources:
   - [Recipe Formatting](https://schema.org/Recipe)
   - [Food Safety](https://www.fda.gov/food/food-safety-modernization-act-fsma)
   - [Cooking Times](https://www.foodsafety.gov/food-safety-charts/safe-minimum-cooking-temperatures)