#%%
from transformer_lens import HookedTransformer
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    default_padding_side="left",
    device=DEVICE
)

snowflake_true = [
    "No two snowflakes are exactly alike.",
    "Snowflakes have a six-sided structure.",
    "Snowflakes form in clouds when water vapor freezes.",
    "Snowflakes start as ice crystals in the atmosphere.",
    "The shape of a snowflake depends on temperature and humidity.",
    "Most snowflakes are less than a centimeter wide.",
    "Snowflakes fall at about 1-6 feet per second.",
    "Snow is composed of many snowflakes clustered together.",
    "Snowflakes can appear clear or white depending on how light reflects.",
    "Snowflakes are made of water that freezes at 32°F (0°C).",
    "Each snowflake has a unique molecular arrangement.",
    "There are many types of snowflakes, including dendrites and plates.",
    "Snowflakes form in hexagonal patterns due to the structure of ice molecules.",
    "The largest snowflake recorded was 15 inches wide.",
    "Snowflakes can form in temperatures as low as -40°F.",
    "A snowflake's symmetry comes from how water molecules arrange themselves.",
    "Snowflakes often have a fern-like pattern called stellar dendrites.",
    "Tiny imperfections in snowflakes can lead to different shapes.",
    "Wet snowflakes stick together more than dry snowflakes.",
    "Snowflakes can reflect sunlight and cause a phenomenon called 'snow blindness.'",
    # "It takes millions of snowflakes to create just one inch of snow.",
    # "Snowflakes often shatter when they hit the ground.",
    # "The study of snowflakes is called crystallography.",
    # "Snowflakes have been studied for over 400 years.",
    # "Some snowflakes are hollow in the center."
]

snowflake_false = [
    "All snowflakes are identical in shape and size.",
    "Snowflakes always have eight sides.",
    "Snowflakes are made of frozen sand particles.",
    "Snowflakes form in the ocean and rise into the clouds.",
    "Snowflakes can only form at temperatures above freezing.",
    "Each snowflake contains a grain of salt at its center.",
    "Snowflakes are created by humans in special labs.",
    "Snowflakes can be square or triangular.",
    "Snowflakes can only fall during the day.",
    "Snowflakes are a type of frozen flower pollen.",
    "Snowflakes grow larger in space than on Earth.",
    "Snowflakes can form in deserts during heatwaves.",
    "All snowflakes are completely flat like paper.",
    "Snowflakes never melt, even in temperatures above freezing.",
    "Snowflakes are heavier than hailstones.",
    "Snowflakes are made entirely of frozen oxygen.",
    "Snowflakes always fall in a straight line to the ground.",
    "The color of a snowflake depends on the moon's phase.",
    "Snowflakes can only form in the Arctic and Antarctic regions.",
    "Snowflakes are a form of frozen lightning.",
    # "Snowflakes glow in the dark due to radioactive properties.",
    # "Snowflakes can speak to each other as they fall.",
    # "Snowflakes can only be seen under a microscope.",
    # "Snowflakes always land upright like tiny pyramids.",
    # "Snowflakes are immune to heat and never evaporate."
]

snowflake_truth = [1] * len(snowflake_true) + [0] * len(snowflake_false)
snowflake_facts = snowflake_true + snowflake_false

prompt_format = "True or False? Fact: {} This fact is: "
#%%
snow_prompts = [prompt_format.format(fact) for fact in snowflake_facts]
snow_toks = model.tokenizer(snow_prompts, padding=True, return_tensors="pt")['input_ids']
# lizard_prompts = [prompt_format.format(fact) for fact in lizard_facts]
# lizard_toks = model.tokenizer(lizard_prompts, padding=True, return_tensors="pt")['input_ids']

# %%
_, snow_cache = model.run_with_cache(snow_toks, names_filter = lambda x: 'resid_pre' in x)
# _, lizard_cache = model.run_with_cache(lizard_toks, names_filter = lambda x: 'resid_pre' in x)
# %% Predict ground truth with cache activations using sklearn logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformer_lens import utils
import numpy as np

for layer in range(model.cfg.n_layers):
    snow_X = utils.to_numpy(snow_cache[f'blocks.{layer}.hook_resid_pre'][:, -1, :])
    snow_y = np.array(snowflake_truth)
    snow_train, snow_test, snow_y_train, snow_y_test = train_test_split(snow_X, snow_y, test_size=0.2, random_state=42)

    # Train logistic regression model
    snow_clf = LogisticRegression(random_state=42).fit(snow_train, snow_y_train)
    print(f'Layer {layer} snowflake fact prediction accuracy: {accuracy_score(snow_y_test, snow_clf.predict(snow_test))}')


# %%
