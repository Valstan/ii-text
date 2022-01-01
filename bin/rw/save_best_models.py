def save_best_models(tuner, x_test, y_test, project_name):
    models = tuner.get_best_models(10)
    for idx, model in enumerate(models):
        model.summary()
        model.evaluate(x_test, y_test)
        model.save('data/' + str(project_name) + '/best_model' + str(idx) + '.h5')

