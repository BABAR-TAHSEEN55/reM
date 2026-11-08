def matches(gesture_result):
    return any(categories and categories[0].category_name == "Open_Palm" for categories in gesture_result.gestures)
