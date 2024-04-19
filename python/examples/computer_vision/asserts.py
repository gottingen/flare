import os


def get_assert_path():

    cwd_path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists("/../../assets"):
        return cwd_path + "/../../assets"

    if os.path.exists("/../../../share/flare/asserts"):
        return cwd_path + "/../../../share/flare/asserts"
    return None
