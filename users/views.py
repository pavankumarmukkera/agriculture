from django.shortcuts import render
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.shortcuts import render,HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'userregistration.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'userregistration.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def imageprediction(request):
    from django.conf import settings
    if request.method=='POST':
        image_file = request.FILES['file']
        # let's check if it is a csv file
        # if not image_file.name.endswith('.png'):
        #     messages.error(request, 'THIS IS NOT A PNG  FILE')
        fs = FileSystemStorage(location="media/rice_test/")
        filename = fs.save(image_file.name, image_file)
        file = settings.MEDIA_ROOT + '//' + 'rice_test' + '//' + filename
        print(file)
    
      
        # detect_filename = fs.save(image_file.name, image_file)

        fileinput = "/media/" + "/rice_test/" +  filename
     
        uploaded_file_url = "/media/" + 'output.jpg'
      # fs.url(filename)
          # fs.url(filename)
        print("Image path ", uploaded_file_url)
        disease_detected = "Tomato_Early_blight"
        from .utility.diease_pesticides import prediction
        from .utility.diease_pesticides import recommendation

        result = prediction(file)
        prevention,__ = recommendation(result)
      
        return render(request, "users/prediction.html", {'y_pred': result,'prevention':prevention,'pesticide':__,'path':fileinput})
    else:
        return render(request, "users/prediction.html",{})
    
def prediction(request):
    import tensorflow
    if request.method=='POST':
        from django.core.files.storage import FileSystemStorage
        from tensorflow.keras.models import load_model
        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/Soil types/")
        filename = fs.save(image_file.name, image_file)
        file = settings.MEDIA_ROOT + '\\' + 'Soil types' + '\\' + filename
        
        # detect_filename = fs.save(image_file.name, image_file)
        fileinput = "/media/" + "/rice_test/" +  filename
        uploaded_file_url = "/media/Soil types/" + filename  # fs.url(filename)
        print("Image path ", uploaded_file_url)
        model_path = settings.MEDIA_ROOT + '\\'+ 'models'+ '\\' + 'model.h5'
        SoilNet = load_model(model_path)
        from .utility.Algorithm import model_predict
        prediction,output_page = model_predict(file,SoilNet)
        soil = prediction.split(':-')[0]
        crops = prediction.split(':-')[1]
        print(soil)
        print("Result=", prediction)
        return render(request, "users/testform.html", {'soil': soil,'crops': crops,'sanju':uploaded_file_url})
    else:
        return render(request, "users/testform.html",{})
    


    


