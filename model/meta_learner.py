def reptile_update(meta_model, task_model, epsilon=0.1):
    new_weights = []
    meta_weights = meta_model.get_weights()
    task_weights = task_model.get_weights()
    for m_w, t_w in zip(meta_weights, task_weights):
        new_weights.append(m_w + epsilon * (t_w - m_w))
    meta_model.set_weights(new_weights)
