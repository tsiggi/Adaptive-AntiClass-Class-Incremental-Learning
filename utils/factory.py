def get_model(model_name, args):
    name = model_name.lower()
    if name == "der":
        from models.der import DER
        return DER(args)
    else:
        raise NotImplementedError("Unknown model {}".format(model_name))
