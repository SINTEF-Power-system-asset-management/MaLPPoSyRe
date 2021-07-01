import nox

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'malpposyre',
                '--max-line-length=127',
                '--count')

@nox.session(python="3.8")
def black(session):
        args = session.posargs or locations
            session.install("black")
                session.run("black", *args)
