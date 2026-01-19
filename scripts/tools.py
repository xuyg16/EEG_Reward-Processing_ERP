def get_valid_input(prompt, valid_options):
    """
    Prompt the user for input until a valid option is provided.

    :param prompt: The prompt message to display to the user
    :param valid_options: A list of valid options
    :return: The valid input provided by the user
    """
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() in [opt.lower() for opt in valid_options]:
            print(f'Input accepted:{user_input}')
            return user_input
        else:
            print(f"Invalid input. Please choose from {valid_options}.")