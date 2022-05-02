from pandas import read_csv
import pytesseract
import numpy as np
import os


def get_user_directories(root_directories):
    user_directories = []
    for directory in root_directories:
        user_directories_parent = os.path.join(
            directory, directory, "Corpus", "Corpus")
        user_directories += list(os.path.join(user_directories_parent, x)
                                 for x in os.listdir(user_directories_parent))
    user_directories = sorted(user_directories, key=lambda x: int(
        x.replace("/", "\\").split("\\")[-1][1:]))

    return user_directories


def get_ad_directories(root_directories):
    ad_directories = []

    for directory in root_directories:
        ad_directories_parent = os.path.join(
            directory, directory, "Ads", "Ads")

        ad_directories += list(os.path.join(ad_directories_parent, x)
                               for x in os.listdir(ad_directories_parent))

    ad_directories = sorted(ad_directories, key=lambda x: int(
        x.replace("/", "\\").split("\\")[-1]))

    return ad_directories


def user_iterator(root_directories):
    user_directories = get_user_directories(root_directories)
    for user_directory in user_directories:
        user_id = user_directory.replace("/", "\\").split("\\")[-1]
        b5 = read_csv(os.path.join(user_directory,
                      f"{user_id}-B5.csv"), delimiter=";")["Answer"].values

        inf = read_csv(os.path.join(
            user_directory, f"{user_id}-INF.csv"), delimiter=";")
        gender_age_income = np.array([[1 if x[0] == "F" else 0 if x[0] == "M" else 0.5, *x[1:]]
                                     for x in inf[["Gender", "Age", "Income"]].values][0])

        rt = read_csv(os.path.join(user_directory,
                      f"{user_id}-RT.csv"), delimiter=";")
        mean_rt_per_cat = np.mean(np.array(
            list(list(int(i) for i in x.split(",")) for x in rt.iloc[1].values)), axis=1)
        ratings = np.array(
            list(map(int, ",".join(rt.iloc[1].values).split(","))))

        yield (user_id, b5, gender_age_income, mean_rt_per_cat, ratings)


def ad_category_iterator(root_directories):
    ad_directories = get_ad_directories(root_directories)
    num_categories = len(ad_directories)
    for ad_directory in ad_directories:
        ad_category_id = ad_directory.replace("/", "\\").split("\\")[-1]
        img_paths = list(os.path.join(ad_directory, x)
                         for x in os.listdir(ad_directory))
        category = int(ad_directory.replace("/", "\\").split("\\")[-1])-1
        category_one_hot = np.zeros(num_categories)
        category_one_hot[category] = 1
        texts = list(map(pytesseract.image_to_string, img_paths))
        yield (ad_category_id, category_one_hot, texts)
