import logging



def traceback_no_record_found_in_sql():
    """
    Log une erreur si aucun enregistrement n'est trouvé dans Cloud SQL.
    """
    logging.error("""
        Désolé, nous n'avons trouvé aucune information correspondant à votre demande.
        Veuillez vérifier votre saisie ou contacter le support pour obtenir de l'aide.
    """)


def traceback_no_urls_retrieved():
    """
    Log une erreur si aucune URL source n'est récupérée pour une question.
    """
    logging.error("""
        Désolé, nous n'avons pas pu récupérer les informations nécessaires pour répondre à votre question.
        Veuillez réessayer ou contacter le support si le problème persiste.
    """)


