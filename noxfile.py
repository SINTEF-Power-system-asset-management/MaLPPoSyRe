import nox


nox.options.sessions = ["lint"]  # Only run black as default
locations = ["malpposyre", "noxfile.py"]


@nox.session()
def lint(session):
    session.install("flake8", "flake8-black")
    session.run("flake8", "malpposyre", "--count")


@nox.session()
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
