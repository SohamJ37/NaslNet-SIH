from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField
from wtforms.validators import DataRequired, URL
from flask_ckeditor import CKEditorField


class PostForm(FlaskForm):
    title = StringField(label='Title', validators=[DataRequired(message="Can't be empty")])

    subtitle = StringField(label='Subtitle', validators=[DataRequired(message="Can't be empty")])

    body = CKEditorField(label='Body', validators=[DataRequired(message="Can't be empty")])

    submit = SubmitField('Submit')


class RegisterForm(FlaskForm):
    name = StringField(label="Name", validators=[DataRequired()])
    email = StringField(label="Email", validators=[DataRequired()])
    password = PasswordField(label="Password", validators=[DataRequired()])
    submit = SubmitField('Submit')


class LoginForm(FlaskForm):
    email = StringField(label="Email", validators=[DataRequired()])
    password = PasswordField(label="Password", validators=[DataRequired()])
    submit = SubmitField("Let Me In")