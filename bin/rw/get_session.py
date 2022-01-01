import json
import os

from bin.rw.get_mongo_base import get_mongo_base


def get_session(name_base, name_collection, name_session, path_logpass='logpass.json'):
    collection = get_mongo_base()[name_collection]
    session = collection.find_one({'title': name_collection})
    session.update(session['config_bases'][name_base])
    del session['config_bases']
    session.update({"name_session": name_session})
    with open(os.path.join(path_logpass), 'r', encoding='utf-8') as f:
        lp = json.load(f)
    session.update(lp)
    return session
