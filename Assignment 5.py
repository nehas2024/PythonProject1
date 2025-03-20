def calculate_grade(score):
    if score >= 90:
        return'your grade is A'
    elif score >= 80 and score < 90:
        return'your grade is B'
    elif score >= 70 and score < 80:
        return'your grade is C'
    elif score >= 60 and score < 70:
        return'your grade is D'
    else:
        return'your grade is E'
score = int(input("enter your score: "))
grade = calculate_grade(score)
print(grade)
