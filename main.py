from app.rag.rag_huggingface import rag_huggingface


def init(query: str) -> list:
    res = rag_huggingface(query)
    return res

# from app.rag.rag_openai import rag_openai
#
#
# def init(query: str) -> list:
#     res = rag_openai(query)
#     return res

if __name__ == '__main__':
    query_list = [
        "What does XAFS stand for and what does it measure in materials?",
        "Why is XAFS considered an important tool in various scientific fields such as biology and materials science?",
        "What technological advancement significantly contributed to the development of XAFS?",
        "What are the main differences between EXAFS and XANES in terms of energy range and theoretical understanding?",
        "How does the photoelectron behave according to the short-range-order theory in XAFS?",
        "What limits the range over which XAFS can provide structural information?",
        "Why was early progress in XAFS slow and what changed to accelerate its development?",
        "What role does the Debye-Waller factor play in the XAFS spectrum?",
        "How is structural information such as interatomic distances extracted from EXAFS data?",
        "What ey theoretical improvements have enhanced the accuracy of EXAFS analysis over time?",
    ]
    for query in query_list:
        print(f"Query: {query}")
        res = init(query)
        print(f"Number of texts: {len(res)}\n")

    # random_query_list = [
    #     "If you could travel anywhere in the world right now, where would you go and why?",
    #     "What do you think life will be like 50 years from now?",
    #     "Which do you prefer: reading books or watching movies, and why?",
    #     "If you could have any superpower, what would it be and how would you use it?",
    #     "What is one skill you’ve always wanted to learn but haven’t had the chance to yet?"
    # ]
    # for query in random_query_list:
    #     print(f"Query: {query}")
    #     res = init(query)
    #     print(f"Number of texts: {len(res)}\n")

    # likely_query = [
    #     "How does the uncertainty principle affect the energy levels of electrons in a hydrogen atom?",
    #     "What are the thermodynamic conditions required for a substance to undergo a phase transition from liquid to gas?",
    #     "How does the structure of a benzene ring influence its chemical reactivity in electrophilic substitution reactions?",
    #     "Why do noble gases have such low boiling points compared to other elements in the periodic table?",
    #     "What is the role of catalysts in lowering the activation energy of a chemical reaction?"
    # ]
    # for query in likely_query:
    #     print(f"Query: {query}")
    #     res = init(query)
    #     print(f"Number of texts: {len(res)}\n")