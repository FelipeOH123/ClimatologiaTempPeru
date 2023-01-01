from datetime import date, datetime, timedelta


def get_first_date_of_current_month(year, month):
    first_date = datetime(year, month, 1)
    return first_date.strftime("%Y-%m-%d")


def get_last_date_of_month(year, month):
    if month == 12:
        last_date = datetime(year, month, 31)
    else:
        last_date = datetime(year, month + 1, 1) + timedelta(days=-1)

    return last_date.strftime("%Y-%m-%d")