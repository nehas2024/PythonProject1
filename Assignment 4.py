score = int(input("enter your score: "))
if score>=90:
    print("your grade is A")
elif score>=80 and score<90:
    print("your grade is B")
elif score>=70 and score<80:
    print("your grade is C")
elif score>=60 and score<70:
    print("your grade is D")
else:
    print("your grade is E")