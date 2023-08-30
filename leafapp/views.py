from django.shortcuts import render, HttpResponse
from . import leaf as l
import os
import uuid

# Create your views here.
def index(request):
    return render(request,'index.html')

def lfprdct(request):
    if request.method =="POST":
        file = request.FILES['file']
        file_extension = os.path.splitext(file.name)[1]  # Get the file extension
        file_name = str(uuid.uuid4()) + file_extension  # Generate a unique file name
        file_path = os.path.join(r'D:\leafproject\temparary files', file_name)  # Set the file path
        
        with open(file_path, 'wb') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        result=l.prdct(file_path)
        return render(request, "index.html",{'result': result})
    return render(request,'index.html')
    