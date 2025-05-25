import re
import pandas as pd

def preprocess(data):
    # WhatsApp date-time pattern (support both 24hr and 12hr with AM/PM)
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s[APap][Mm])?\s-\s'

    # Split messages and extract dates from the raw chat text
    messages = re.split(pattern, data)[1:]  # First split is empty before first date
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Clean unicode space characters in date strings
    df['message_date'] = df['message_date'].str.replace('\u202f', ' ', regex=False).str.replace('\xa0', ' ', regex=False)

    # Parse dates: try 24hr format first, then 12hr AM/PM, else auto parse
    try:
        df['date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ', errors='raise')
    except Exception:
        try:
            df['date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ', errors='raise')
        except Exception:
            df['date'] = pd.to_datetime(df['message_date'], errors='coerce')

    # Extract user and message text
    users = []
    messages_clean = []

    # Regex to split on the first occurrence of "user: " pattern
    # Allows user names with spaces, emojis, special chars (non-colon until first colon)
    user_message_pattern = re.compile(r'^([^:]+):\s(.*)', re.DOTALL)

    for message in df['user_message']:
        match = user_message_pattern.match(message)
        if match:
            users.append(match.group(1).strip())
            messages_clean.append(match.group(2).strip())
        else:
            # No user found means system/group notification
            users.append('group_notification')
            messages_clean.append(message.strip())

    df['user'] = users
    df['message'] = messages_clean

    # Drop original columns not needed anymore
    df.drop(columns=['user_message', 'message_date'], inplace=True)

    # Extract datetime features for analysis
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create hourly period like "23-00"
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(f"{hour}-{hour + 1}")

    df['period'] = period

    return df
