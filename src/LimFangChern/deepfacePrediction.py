def deepfacePrediction(face):
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)

        if isinstance(result, list) and len(result) > 0:
            age = str(result[0]['age'])
            gender = str(result[0]['gender'])
        else:
            age = "unknown"
            gender = "unknown"
    except Exception as e:
        print(f"Error during DeepFace prediction: {e}")
        age = "unknown"
        gender = "unknown"

    return age, gender
