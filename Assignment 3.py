age = int(input("Enter a  patient age: "))
if age>=18:
    print("The medicine can be given")
elif age>=15 and age<=18:
    weight = int(input("Enter a weight in kg: "))
    if weight>=55:
        print("The medicine can be given")
    else:
        print("The medicine cannot be given")
else:
    print("The medicine cannot be given")


