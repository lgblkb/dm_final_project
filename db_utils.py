import contextlib

import sqlalchemy as sa
from dynaconf import settings
from lgblkb_tools import logger
from sqlalchemy import orm
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker

engine = sa.create_engine(URL('postgresql', **settings.POSTGIS), echo=False)
Session = sessionmaker(bind=engine)


class ModelUtils(object):
    
    def save(self):
        raise NotImplementedError
    
    def reconnect(self):
        raise NotImplementedError
    
    def query_get(self, model_class, *primary_keys):
        raise NotImplementedError
    
    def query_get_save(self, model_class, *primary_keys):
        raise NotImplementedError


@contextlib.contextmanager
def db_session():
    session: orm.Session = Session()
    
    try:
        yield session
    except Exception as exc:
        logger.debug("exc: %s", exc)
        session.rollback()
        raise
    else:
        session.commit()
        session.close()


def main():
    logger.debug("settings.POSTGIS:\n%s", settings.POSTGIS)
    pass


if __name__ == '__main__':
    main()
