from server import db, create_app
from server.models import User
from werkzeug.security import generate_password_hash, check_password_hash

app = create_app()


def print_users():
    print(":")
    users = User.query.all()
    for user in users:
        print(user.name, user.email)


def add_user(name, email, password):
    user = User.query.filter_by(email=email).first()
    if user:  # if a user is found, return
        print("user with email ", email, " already exists.")
        return

    # create a new user.
    new_user = User(email=email, name=name,
                    password=generate_password_hash(password, method='sha256'))

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    print("updated db.")
    return print_users()


def delete_user(email):
    user = User.query.filter_by(email=email).first()
    if not user:  # if the user is not found, return
        print("user with email ", email, " does not exist.")
        return

    # delete from the database
    db.session.delete(user)
    db.session.commit()

    print("updated db.")
    return print_users()


import sys
with app.app_context():
    # print db
    print_users()
    # add user
    add_user(name="test3", email="test3@gog.com", password="heyJude!")
    # remove user
    delete_user(email="test3@gog.com")
    #print("adding user email",sys.argv[3])
    for em in sys.argv[1:]:
        delete_user(email=em)
    #add_user(name=sys.argv[2], email=sys.argv[3],password=sys.argv[1])
