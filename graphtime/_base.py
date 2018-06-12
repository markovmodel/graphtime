from future import absolute_import


class Model(object):
  def set_model_parameters(self, **params):
    for p in params:
      setattr(self, p, params[p])


