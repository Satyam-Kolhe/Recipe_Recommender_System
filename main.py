import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from scipy.sparse import csr_matrix
import numpy as np
import lzma

import gzip
import shutil
with lzma.open('df_dict(1).pkl.xz', 'rb') as f:
    df_dict = pickle.load(f)

cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
vectors = pickle.load(open('vectors.pkl', 'rb'))
# df_dict = pickle.load(open('df_dict.pkl', 'rb'))
DataFrame = pd.DataFrame(df_dict)
ingredients_lst = DataFrame['ingredients_str'].unique().tolist()


# Flatten the ingredients from your DataFrame into a single list of unique ingredients
def get_unique_ingredients(DataFrame):
    all_ingredients = DataFrame['ingredients_list'].explode().unique().tolist()
    return sorted(all_ingredients)  # Sorted for better user experience

# Get a list of unique ingredients
unique_ingredients = get_unique_ingredients(DataFrame)

# Function to filter ingredients based on user input
def filter_ingredients(query):
    return [ingredient for ingredient in unique_ingredients if query.lower() in ingredient.lower()]

# Initialize session state for selected ingredients
if 'selected_ingredients' not in st.session_state:
    st.session_state['selected_ingredients'] = []

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image('logo for Flavors Finder.png', width=300)

st.markdown("<h1 style='text-align: center;'>Flavors Finder</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Your Recipe Recommender System</h2>", unsafe_allow_html=True)

user_ingredient_query = st.text_input('Enter Ingredient Name')

# Display filtered ingredients in a multiselect box
filtered_ingredients = filter_ingredients(user_ingredient_query)

selected_ingredients = st.multiselect('Select Ingredients', filtered_ingredients)

if st.button('Add Ingredient'):
    for ingredient in selected_ingredients:
        if ingredient not in st.session_state['selected_ingredients']:
            st.session_state['selected_ingredients'].append(ingredient)

# Beautify the display of selected ingredients with remove option
if st.session_state['selected_ingredients']:
    st.markdown("### Selected Ingredients")
    st.markdown('<div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">', unsafe_allow_html=True)
    for ingredient in st.session_state['selected_ingredients']:
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f'<div style="color: #4CAF50; font-size: 16px; margin: 5px 0;">• {ingredient}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("➖", key=f'remove_ingredient_{ingredient}', help="Click to remove this ingredient"):
                st.session_state['selected_ingredients'].remove(ingredient)
    st.markdown("</div>", unsafe_allow_html=True)

UNSPLASH_ACCESS_KEY = '8CSZZy6w2oilftllf0kIlxz45PR4wthG2WOO6JYW2CY'

# Function to fetch image URLs from Unsplash
def fetch_image_url(recipe_name):
   query = f"{recipe_name} fooddish"
   url = f'https://api.unsplash.com/search/photos?query={query}&client_id={UNSPLASH_ACCESS_KEY}'
   response = requests.get(url)
   data = response.json()
   if data['results']:
      return data['results'][0]['urls']['regular']
   return "https://via.placeholder.com/150"  # Fallback image URL

# Function to suggest recipes based on user-provided ingredients
def suggest_recipes(user_ingredients):
    input_ingredients_str = ' '.join([ingredient.replace(' ', '') for ingredient in user_ingredients])
    input_vector = cv.transform([input_ingredients_str])

    # Compute similarity scores
    similarity_scores = cosine_similarity(input_vector, vectors).flatten()

    DataFrame['similarity'] = similarity_scores

    # Check for recipes matching the ingredients
    top_recipes = DataFrame.nlargest(10, 'similarity')[['name', 'minutes', 'steps', 'ingredients']]
    if not top_recipes.empty:
        top_recipes['image_url'] = top_recipes['name'].apply(fetch_image_url)
        top_recipes['name'] = top_recipes['name'].apply(lambda x: x.title())
        return top_recipes
    return top_recipes

if st.button("Recommend Recipes"):
    if not st.session_state['selected_ingredients']:
        st.error("Please select some ingredients.")
    else:
        recommendations = suggest_recipes(st.session_state['selected_ingredients'])
        st.markdown("### Recommended Recipes")
        if recommendations.empty:
            st.write("No matching recipes found.")
        else:
            for i in range(len(recommendations)):
                st.header(recommendations.iloc[i]['name'])
                st.write('Time to make:', recommendations.iloc[i]['minutes'], 'minutes')
                st.image(recommendations.iloc[i]['image_url'], use_container_width=True)
                # Display ingredients in unordered list
                recipe_ingredients = ast.literal_eval(recommendations.iloc[i]['ingredients'])
                st.markdown("#### Ingredients")
                for ingredient in recipe_ingredients:
                    st.markdown(f"- {ingredient.capitalize()}")
                # Display steps in ordered list
                steps = ast.literal_eval(recommendations.iloc[i]['steps'])
                st.markdown("#### Steps")
                for idx, step in enumerate(steps, 1):
                    st.markdown(f"{idx}. {step.capitalize()}")

