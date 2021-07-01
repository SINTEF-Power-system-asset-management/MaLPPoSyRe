import nox

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'malpposyre',
                '--max-line-length=127',
                '--count')
