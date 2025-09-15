from django.conf import settings

knowledge_base = {
    "Tomato_Early_blight": {
        
            "prevention": [
                'Pruning: Regularly prune infected branches and remove mummified fruit from the tree. This helps to reduce the source of infection.',
                'Sanitation: Maintain good orchard hygiene by cleaning up and destroying fallen leaves and infected plant material. This helps to reduce the overwintering spores.',
                'Resistant Varieties: Consider planting apple varieties that are resistant to Black Rot.'
            ],
            "chemical_controls": [
                'Captan: Captan is a broad-spectrum fungicide effective against Black Rot. It should be applied according to the manufacturers recommendations.',

                'Myclobutanil: Myclobutanil is a systemic fungicide that provides protective and curative action against various fungal diseases, including Black Rot.'
            ]
        
    },
     "Tomato_Early_blight": {
        
            "prevention": [
                'Pruning: Regularly prune infected branches and remove mummified fruit from the tree. This helps to reduce the source of infection.',
                'Sanitation: Maintain good orchard hygiene by cleaning up and destroying fallen leaves and infected plant material. This helps to reduce the overwintering spores.',
                'Resistant Varieties: Consider planting apple varieties that are resistant to Black Rot.'
            ],
            "chemical_controls": [
                'Captan: Captan is a broad-spectrum fungicide effective against Black Rot. It should be applied according to the manufacturers recommendations.',

                'Myclobutanil: Myclobutanil is a systemic fungicide that provides protective and curative action against various fungal diseases, including Black Rot.'
            ]
        
    },
        "Tomato_Early_blight": {
        
            "prevention": [
                'Pruning: Regularly prune infected branches and remove mummified fruit from the tree. This helps to reduce the source of infection.',
                'Sanitation: Maintain good orchard hygiene by cleaning up and destroying fallen leaves and infected plant material. This helps to reduce the overwintering spores.',
                'Resistant Varieties: Consider planting apple varieties that are resistant to Black Rot.'
            ],
            "chemical_controls": [
                'Captan: Captan is a broad-spectrum fungicide effective against Black Rot. It should be applied according to the manufacturers recommendations.',

                'Myclobutanil: Myclobutanil is a systemic fungicide that provides protective and curative action against various fungal diseases, including Black Rot.'
            ]
        
    },
        "Tomato_Early_blight": {
        
            "prevention": [
                'Pruning: Regularly prune infected branches and remove mummified fruit from the tree. This helps to reduce the source of infection.',
                'Sanitation: Maintain good orchard hygiene by cleaning up and destroying fallen leaves and infected plant material. This helps to reduce the overwintering spores.',
                'Resistant Varieties: Consider planting apple varieties that are resistant to Black Rot.'
            ],
            "chemical_controls": [
                'Captan: Captan is a broad-spectrum fungicide effective against Black Rot. It should be applied according to the manufacturers recommendations.',

                'Myclobutanil: Myclobutanil is a systemic fungicide that provides protective and curative action against various fungal diseases, including Black Rot.'
            ]
        
    },
        "Tomato_Early_blight": {
        
            "prevention": [
                'Pruning: Regularly prune infected branches and remove mummified fruit from the tree. This helps to reduce the source of infection.',
                'Sanitation: Maintain good orchard hygiene by cleaning up and destroying fallen leaves and infected plant material. This helps to reduce the overwintering spores.',
                'Resistant Varieties: Consider planting apple varieties that are resistant to Black Rot.'
            ],
            "chemical_controls": [
                'Captan: Captan is a broad-spectrum fungicide effective against Black Rot. It should be applied according to the manufacturers recommendations.',

                'Myclobutanil: Myclobutanil is a systemic fungicide that provides protective and curative action against various fungal diseases, including Black Rot.'
            ]
        
    },
}







def prediction(file):
    from keras.preprocessing import image
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np
    
    new_img = image.load_img(file, target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    model = settings.MEDIA_ROOT + '//' + 'Model.hdf5'
    print("Following is our prediction:")
    model = load_model(model)
    prediction = model.predict(img)
    print(prediction)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    d = prediction.flatten()
    j = d.max()
    li=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Huanglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early blight', 'Potato___Late blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early blight', 'Tomato___Late blight', 'Tomato___Leaf Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    for index,item in enumerate(d):
        if item == j:
            class_name = li[index]

    print(class_name)
    return class_name

def recommendation(disease):
  import json
  file = model = settings.MEDIA_ROOT + '//' + 'pesticides.json'
  with open(file) as f:
    pesticides = json.load(f)
  

  for pesticide in pesticides:
        if pesticide['disease'] == disease:
            prevention = pesticide['preventions']
            pesticide = pesticide['pesticide']
            print(prevention)

            return prevention,pesticide
        
            

# Example usage
