{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# **1. Análise Exploratória de Dados:**"
      ],
      "metadata": {
        "id": "STrbFoT_LGp_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Bibliotecas**"
      ],
      "metadata": {
        "id": "y8AcqUgaMQoQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.pipeline import Pipeline, make_pipeline\n",
        "from sklearn.preprocessing import FunctionTransformer\n",
        "from sklearn.metrics import roc_curve, auc"
      ],
      "metadata": {
        "id": "yXcGMl1wLHB9"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Com objetivo de importar varias bibliotecas com modulos de dados."
      ],
      "metadata": {
        "id": "oihZbBt_LMSV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Carregador de Dados**"
      ],
      "metadata": {
        "id": "LUkhF0hQLHeH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('dataset.csv', delimiter=';')\n",
        "print(df.head())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Q4Y_S8y8H_Hj",
        "outputId": "2980982c-fbca-4dee-f2e5-57460efc4838"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   Transaction ID        Date   Time   Card Type Entry Mode  Amount  \\\n",
            "0               1  19/10/2023  05:31  MasterCard        Tap  391.56   \n",
            "1               2  16/10/2023  12:56  MasterCard        Tap  167.67   \n",
            "2               3  29/09/2023  07:57  MasterCard        PIN  126.24   \n",
            "3               4  13/10/2023  00:08  MasterCard        Tap  496.80   \n",
            "4               5  02/10/2023  23:19  MasterCard        Tap  446.88   \n",
            "\n",
            "  Transaction Type Merchant Group Transaction Country Shipping Address  \\\n",
            "0              POS       Clothing               Italy    61 Redwood St   \n",
            "1           Online       Clothing              France      945 Pine St   \n",
            "2           Online    Electronics               Spain       773 Oak St   \n",
            "3           Online    Electronics                  UK     436 Maple St   \n",
            "4              POS        Grocery              France      887 Pine St   \n",
            "\n",
            "  Billing Address  Gender  Age Issuing Bank Fraudulent  \n",
            "0    726 Maple St  Female   55     XYZ Bank         No  \n",
            "1    252 Cedar St    Male   37     DEF Bank         No  \n",
            "2     177 Main St    Male   53     ABC Bank         No  \n",
            "3     758 Main St    Male   21     DEF Bank         No  \n",
            "4     629 Main St    Male   52     XYZ Bank         No  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Possui o objetivo de ler o arquivo CSV chamado 'dataset.csv' usando a biblioteca pandas em Python."
      ],
      "metadata": {
        "id": "d9J4xE7VLF4C"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Verificação das colunas**"
      ],
      "metadata": {
        "id": "aUHtlPXILg05"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cPLDD8szKpIj",
        "outputId": "a32684ab-0510-42e6-fc43-9890b1779664"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['Transaction ID', 'Date', 'Time', 'Card Type', 'Entry Mode', 'Amount',\n",
            "       'Transaction Type', 'Merchant Group', 'Transaction Country',\n",
            "       'Shipping Address', 'Billing Address', 'Gender', 'Age', 'Issuing Bank',\n",
            "       'Fraudulent'],\n",
            "      dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Exibir os nomes das colunas do DataFrame \"df\"."
      ],
      "metadata": {
        "id": "-PubQXR4Li93"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Verificação de distribuição das classes**"
      ],
      "metadata": {
        "id": "tn3_KpltLjKp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(x='Fraudulent', data=df)\n",
        "plt.title('Distribuição de Classes')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "qgP4-DvPJdHf",
        "outputId": "ed3ce76b-2473-45cf-a657-dc4a1bdc1003"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAHHCAYAAABeLEexAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA1iElEQVR4nO3deVxWZf7/8fcNKKJsoiySJIyWgjGZ2hhq5igjmlamo4ND5jYyY5qDlhpNYtlC2rikZVQzKZk1uZSVjVvuGalZNuaWGeaSIGWASwLC+f3Rl/vnLbgRcN94vZ6Px3k8PNd1nXM+h8dNvDvnOue2WZZlCQAAwGBuzi4AAADA2QhEAADAeAQiAABgPAIRAAAwHoEIAAAYj0AEAACMRyACAADGIxABAADjEYgAAIDxCERADfb444/LZrNVy7E6d+6szp0729fXr18vm82mxYsXV9oxDh48KJvNpnnz5l31tosXL5a/v786dOig/fv3KzExUTNnzqy02i7FZrPp8ccfr5ZjXWjevHmy2Ww6ePCgU44PXCsIRICLKP3DVrrUqVNHoaGhiouL06xZs3Ty5MlKOc7333+vxx9/XDt27KiU/bmKqVOnKjExUY0aNVKLFi30zjvvqHfv3s4uq8KKi4s1d+5cde7cWQEBAfL09FR4eLiGDBmizz77zNnlAdccD2cXAMDR5MmTFRERoaKiImVlZWn9+vVKSkrS9OnT9f777+u3v/2tfexjjz2mRx555Kr2//333+uJJ55QeHi4WrVqdcXbrVq16qqOUxFNmjTRzz//rFq1al31tosWLdJ1110nDw8P5eTkyMfHR3Xq1KmCKqvezz//rD59+mjFihXq1KmTHn30UQUEBOjgwYNauHCh0tPTdejQITVu3NjZpQLXDAIR4GJ69Oihtm3b2teTk5O1du1a9erVS3fffbf27NkjLy8vSZKHh4c8PKr21/jMmTOqW7euateuXaXHkWS/MlYRTZo0sf87MDCwskpyinHjxmnFihWaMWOGkpKSHPomTZqkGTNmOKcw4BrGLTOgBujSpYsmTpyo7777Tm+88Ya9vbw5RKtXr1bHjh3l7+8vb29vNW/eXI8++qikX+b93HrrrZKkIUOG2G/Plc7Z6dy5s2666SZt375dnTp1Ut26de3bXjiHqFRxcbEeffRRhYSEqF69err77rt1+PBhhzHh4eEaPHhwmW0v3OfF5hDt3btX/fv3V2BgoLy8vNS8eXP94x//sPdnZmZqxIgRuvHGG+Xl5aUGDRqoX79+5c6r+fbbb9WvXz8FBASobt26uu222/Thhx+WGVeegoICjRkzRoGBgfLx8dHdd9+tI0eOlDv26NGjGjp0qIKDg+Xp6amWLVvqtddeu+wxjhw5opdffll/+MMfyoQhSXJ3d9fDDz98yatD7733nnr27KnQ0FB5enqqadOmevLJJ1VcXOwwbv/+/erbt69CQkJUp04dNW7cWPHx8crLy7OPudTn6fyfy6RJk9SsWTN5enoqLCxM48ePV0FBgcO4K9kX4CxcIQJqiIEDB+rRRx/VqlWrNHz48HLH7Nq1S7169dJvf/tbTZ48WZ6envrmm2+0efNmSVJkZKQmT56slJQUJSYm6vbbb5cktW/f3r6PH3/8UT169FB8fLzuu+8+BQcHX7Kup59+WjabTRMmTNDx48c1c+ZMxcbGaseOHfYrWb/G//73P91+++2qVauWEhMTFR4ergMHDuiDDz7Q008/LUnasmWLMjIyNGDAADVu3FiZmZlKS0tT586dtXv3btWtW1eSlJ2drfbt2+vMmTMaPXq0GjRooPT0dN19991avHix7r333kvW8pe//EVvvPGG/vznP6t9+/Zau3atevbsWWZcdna2brvtNtlsNo0aNUqBgYFavny5hg0bpvz8/HKDTqnly5fr3LlzGjhwYIV/ZvPmzZO3t7fGjh0rb29vrV27VikpKcrPz9dzzz0nSSosLFRcXJwKCgr04IMPKiQkREePHtWyZcuUm5srPz+/y36eJKmkpER33323Pv74YyUmJioyMlI7d+7UjBkz9PXXX2vp0qWSLv/ZBJzOAuAS5s6da0mytm3bdtExfn5+1i233GJfnzRpknX+r/GMGTMsSVZOTs5F97Ft2zZLkjV37twyfXfccYclyUpLSyu374477rCvr1u3zpJkXXfddVZ+fr69feHChZYk6/nnn7e3NWnSxBo0aNBl95mZmVmmtk6dOlk+Pj7Wd99957BtSUmJ/d9nzpwps++MjAxLkvX666/b25KSkixJ1qZNm+xtJ0+etCIiIqzw8HCruLi4zH5K7dixw5JkPfDAAw7tf/7zny1J1qRJk+xtw4YNsxo1amT98MMPDmPj4+MtPz+/custNWbMGEuS9cUXX1x0zPlKPzeZmZn2tvL2/9e//tWqW7eudfbsWcuyLOuLL76wJFmLFi266L6v5PM0f/58y83NzeFnalmWlZaWZkmyNm/efMX7ApyJW2ZADeLt7X3Jp838/f0l/XLLpKSkpELH8PT01JAhQ654/P333y8fHx/7+h//+Ec1atRI//3vfyt0/PPl5ORo48aNGjp0qK6//nqHvvNvFZ5/JaqoqEg//vijmjVrJn9/f33++ef2vv/+97/63e9+p44dO9rbvL29lZiYqIMHD2r37t0XraX0fEaPHu3QfuHVHsuytGTJEt11112yLEs//PCDfYmLi1NeXp5DTRfKz8+XJIef6dU6/+dx8uRJ/fDDD7r99tt15swZ7d27V5Lk5+cnSVq5cqXOnDlT7n6u5PO0aNEiRUZGqkWLFg7n2qVLF0nSunXrrnhfgDMRiIAa5NSpU5f8Q/mnP/1JHTp00F/+8hcFBwcrPj5eCxcuvKo/QNddd91VTaC+4YYbHNZtNpuaNWtWKe/F+fbbbyVJN9100yXH/fzzz0pJSVFYWJg8PT3VsGFDBQYGKjc312E+zHfffafmzZuX2T4yMtLefzHfffed3Nzc1LRpU4f2C/eXk5Oj3NxcvfLKKwoMDHRYSoPm8ePHL3ocX19fSfpVr1nYtWuX7r33Xvn5+cnX11eBgYG67777JMn+84iIiNDYsWP1r3/9Sw0bNlRcXJxefPFFh5/XlXye9u/fr127dpU51xtvvNHhXCvjswlUJeYQATXEkSNHlJeXp2bNml10jJeXlzZu3Kh169bpww8/1IoVK/T222+rS5cuWrVqldzd3S97nMqY93Ohi708sri4+IpqupwHH3xQc+fOVVJSkmJiYuTn5yebzab4+Phq/4Nberz77rtPgwYNKnfM+a9OuFCLFi0kSTt37ryq1yKUys3N1R133CFfX19NnjxZTZs2VZ06dfT5559rwoQJDj+PadOmafDgwXrvvfe0atUqjR49Wqmpqfr000/VuHHjK/o8lZSUKDo6WtOnTy+3nrCwMEmV89kEqhKBCKgh5s+fL0mKi4u75Dg3Nzd17dpVXbt21fTp0/XMM8/oH//4h9atW6fY2NhKf7P1/v37HdYty9I333zj8Ee/fv36ys3NLbPtd999p9/85jcX3Xdp31dffXXJGhYvXqxBgwZp2rRp9razZ8+WOWaTJk20b9++MtuX3kY6/9H9CzVp0kQlJSU6cOCAw1WhC/dX+gRacXGxYmNjL1l3eXr06CF3d3e98cYbFZpYvX79ev34449655131KlTJ3t7ZmZmueOjo6MVHR2txx57TJ988ok6dOigtLQ0PfXUU5Iu/3lq2rSpvvzyS3Xt2vWyn63L7QtwJm6ZATXA2rVr9eSTTyoiIkIJCQkXHXfixIkybaVXGUofga5Xr54klRtQKuL11193uL2zePFiHTt2TD169LC3NW3aVJ9++qkKCwvtbcuWLSvzeP6FAgMD1alTJ7322ms6dOiQQ59lWfZ/u7u7O6xL0uzZs8s8Zn7nnXdq69atysjIsLedPn1ar7zyisLDwxUVFXXRWkrPZ9asWQ7tF349iLu7u/r27aslS5aUG+RycnIuegzplysqw4cP16pVqzR79uwy/SUlJZo2bdpFH/cvvdJy/s+jsLBQc+bMcRiXn5+vc+fOObRFR0fLzc3N/lm5ks9T//79dfToUb366qtlxv788886ffr0Fe8LcCauEAEuZvny5dq7d6/OnTun7OxsrV27VqtXr1aTJk30/vvvX/LFhZMnT9bGjRvVs2dPNWnSRMePH9ecOXPUuHFj+0Tipk2byt/fX2lpafLx8VG9evXUrl07RUREVKjegIAAdezYUUOGDFF2drZmzpypZs2aObwa4C9/+YsWL16s7t27q3///jpw4IDeeOONMvNxyjNr1ix17NhRrVu3VmJioiIiInTw4EF9+OGH9q8f6dWrl+bPny8/Pz9FRUUpIyNDH330kRo0aOCwr0ceeURvvfWWevToodGjRysgIEDp6enKzMzUkiVL5OZ28f9HbNWqlQYMGKA5c+YoLy9P7du315o1a/TNN9+UGfvss89q3bp1ateunYYPH66oqCidOHFCn3/+uT766KNyw8H5pk2bpgMHDmj06NF655131KtXL9WvX1+HDh3SokWLtHfvXsXHx5e7bfv27VW/fn0NGjRIo0ePls1m0/z588sExrVr12rUqFHq16+fbrzxRp07d07z58+3Bzrpyj5PAwcO1MKFC/W3v/1N69atU4cOHVRcXKy9e/dq4cKFWrlypdq2bXtF+wKcyolPuAE4T+nj06VL7dq1rZCQEOsPf/iD9fzzzzs82l7qwsfu16xZY91zzz1WaGioVbt2bSs0NNQaMGCA9fXXXzts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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Esses comandos geram e exibem um gráfico de contagem das classes da variável 'Fraudulent' no DataFrame usando as bibliotecas seaborn e matplotlib.pyplot. O gráfico oferece uma visão rápida da distribuição das classes no conjunto de dados."
      ],
      "metadata": {
        "id": "2huGEPPjp5lw"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Identificador de possíveis desequilíbrios**"
      ],
      "metadata": {
        "id": "BcRWyyjzL8VJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "frauds = df[df['Fraudulent'] == 1]\n",
        "legitimate = df[df['Fraudulent'] == 0]"
      ],
      "metadata": {
        "id": "Vy8b2WjtK4uE"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Estes comandos criam dois novos DataFrames, frauds e legitimate, que contêm apenas as linhas do DataFrame original onde a coluna 'Fraudulent' é igual a 1 (fraudes) e 0 (não fraudes), respectivamente. Essa separação facilita a análise e manipulação específica de cada classe em conjuntos de dados desbalanceados."
      ],
      "metadata": {
        "id": "8cLY0b7aMB2G"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Exploração de padrões nos dados**"
      ],
      "metadata": {
        "id": "b0_igYdrMB5P"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "correlation_matrix = df.corr()\n",
        "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')\n",
        "plt.title('Matriz de Correlação')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 506
        },
        "id": "riSVe6ReK6KZ",
        "outputId": "6599ac94-64b1-4989-ae53-0bc192b895d8"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-7-877988f8955a>:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.\n",
            "  correlation_matrix = df.corr()\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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R2P06NElrjt06NBTt12zZs1zHZPJABERkRY+qIiIiIiMClsGiIiItBhqAKGhMBkgIiLSos/7DFQG7CYgIiIycmwZICIi0qJiNwEREZFx42wCIiIiMipsGSAiItLC2QRERERGzpB3IDQEJgNERERajK1lgGMGiIiIjBxbBoiIiLQY22wCJgNERERajO0+A+wmICIiMnJsGSAiItJibAMImQwQERFp4YOKiIiIyKiwZYCIiEiLsQ0gZDJARESkhWMGDEQiZY8FPbZwwCZDh0AVyJj1fQ0dAlU0a68YOoJXSoVJBoiIiCoKtgwQEREZORXvQEhERGTcjK1lgB31RERERo4tA0RERFqMrWWAyQAREZEWY7vPALsJiIiIjBxbBoiIiLQIziYgIiIybsY2ZoDdBEREREaOLQNERERajG0AIZMBIiIiLewmICIiIqPClgEiIiItxtYywGSAiIhIC8cMEBERGTljaxngmAEiIiIjx5YBIiIiLSqVoSN4uZgMEBERaWE3ARERERkVtgwQERFpMbaWASYDREREWoxtaiG7CYiIiIwcWwaIiIi0CL32E0j0uK/ywWSAiIhIi7GNGWA3ARERkZFjywAREZEW3nSIiIjIyBlbNwGTASIiIi2cWkhERERGhS0DREREWthNQEREZOSEXvsJKv59BthNQEREZOTYMkBERKTF2AYQMhkgIiLSYmxjBthNQEREZOTYMkBERKRFZWT9BEwGiIiItLCbgIiIiIwKWwaIiIi0GFvLAJMBIiIiLSojywbYTUBERKRFqPS36GrJkiXw8vKCmZkZ/Pz8cPr06VLrrlixAm3btoW9vT3s7e3h7+//1PqlYTJARERUQWzatAnjxo1DSEgIzp07h8aNG6NTp064f/9+ifUPHTqE/v374+DBgzhx4gQ8PT3x9ttv4+7duzodl8kAERGRFiGE3hZdhIWFYcSIERg6dCjq1auH8PBwWFhYYNWqVSXW//XXXzFq1Cg0adIEderUwU8//QSVSoWIiAidjssxA0RERFpUz9G8X5q8vDzk5eVplCkUCigUCo2y/Px8nD17FsHBweoyqVQKf39/nDhxokzHys7ORkFBARwcHHSKkS0DRERE5Sg0NBS2trYaS2hoaLF6ycnJUCqVcHFx0Sh3cXFBQkJCmY41ceJEuLu7w9/fX6cY2TJARESkRdfm/aeZFByM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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        " Os comandos geram e exibem um mapa de calor da matriz de correlação das variáveis no DataFrame df. O mapa de calor destaca visualmente as relações de correlação entre as diferentes características do conjunto de dados, proporcionando uma visão rápida dos padrões de associação entre as variáveis."
      ],
      "metadata": {
        "id": "4vJR5VCOMG_2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **2. Pré-processamento de Dados:**\n",
        "\n"
      ],
      "metadata": {
        "id": "buACDY3PMHbI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Exibir as colunas categóricas e data a serem tratadas**"
      ],
      "metadata": {
        "id": "51V6rBPkMoj-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "categorical_columns = ['Card Type', 'Entry Mode', 'Transaction Type', 'Merchant Group', 'Transaction Country', 'Issuing Bank']\n",
        "date_columns = ['Date']\n",
        "time_columns = ['Time']"
      ],
      "metadata": {
        "id": "42XQV0sIQLdX"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Criar listas que categorizam as colunas de um DataFrame em diferentes tipos, para facilitar a manipulação e análise específica desses tipos de dados."
      ],
      "metadata": {
        "id": "taOcBLMlrAWb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Divisão dos dados em conjuntos de treinamento, validação e teste**"
      ],
      "metadata": {
        "id": "rGFoR2WsrHBs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.drop('Fraudulent', axis=1)\n",
        "y = df['Fraudulent']\n",
        "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)\n",
        "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)"
      ],
      "metadata": {
        "id": "zGOHPwI5RUCW"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Criação de conjuntos de treinamento, validação e teste a partir de um DataFrame, facilitando a implementação e avaliação de modelos de machine learning."
      ],
      "metadata": {
        "id": "3r7WLSao0pdT"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Tratamento de colunas categóricas**"
      ],
      "metadata": {
        "id": "HyeWf_zQMrFu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)"
      ],
      "metadata": {
        "id": "rSKmFvvyRukU"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "O comando cria um transformador chamado categorical_transformer usando OneHotEncoder para codificar variáveis categóricas com abordagem one-hot, ignorando valores desconhecidos e produzindo uma matriz densa como resultado."
      ],
      "metadata": {
        "id": "P_g2nXl92KW5"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Tratamento de colunas não categóricas**"
      ],
      "metadata": {
        "id": "Qynpe5OE2Ken"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "non_categorical_transformer = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='median')),  # Preencher valores ausentes com a mediana\n",
        "    ('scaler', StandardScaler())  # Padronizar os valores\n",
        "])"
      ],
      "metadata": {
        "id": "oqWSWI96SoVZ"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Com o objetivo de criar um transformador chamado non_categorical_transformer usando uma pipeline do scikit-learn. Ele preenche valores ausentes com a mediana e padroniza as variáveis numéricas."
      ],
      "metadata": {
        "id": "WlIMYDQO2uVT"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Aplicação de transformações**"
      ],
      "metadata": {
        "id": "Erb59SYM2uv-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('cat', categorical_transformer, categorical_columns),\n",
        "        ('non_cat', non_categorical_transformer, X_train.select_dtypes(exclude='object').columns)\n",
        "    ]\n",
        ")\n"
      ],
      "metadata": {
        "id": "wm9UnFPzZSqe"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Possui o objetivo de criar um transformador chamado preprocessor usando ColumnTransformer do scikit-learn. Ele aplica transformações específicas a diferentes conjuntos de colunas em um conjunto de dados."
      ],
      "metadata": {
        "id": "gsBu-O6i499l"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Modelo**"
      ],
      "metadata": {
        "id": "kysWHqyYMVCy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=10, random_state=42)"
      ],
      "metadata": {
        "id": "LugEvMe9ZVc0"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Possui o objetivo de criar um modelo de classificação baseado em uma MLP com uma camada oculta de 64 neurônios, treinado por até 10 épocas, e garantindo reprodutibilidade ao definir a semente aleatória."
      ],
      "metadata": {
        "id": "wNxN7PbNMQjq"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Pipeline completo**"
      ],
      "metadata": {
        "id": "mwfzMuE5MUnb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "clf = Pipeline(steps=[('preprocessor', preprocessor),\n",
        "                      ('classifier', model)])"
      ],
      "metadata": {
        "id": "Yf5l36rpZqps"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Criação de um pipeline no scikit-learn, que é uma maneira de encadear vários passos de processamento de dados e modelagem em um único objeto."
      ],
      "metadata": {
        "id": "FcRL50xNNNrM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Treinamento do modelo**"
      ],
      "metadata": {
        "id": "0GxGYqdINNxD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "clf.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 310
        },
        "id": "LzoOTpgvZt23",
        "outputId": "31e1e0c6-9a7a-47d1-da4b-4686e628b71c"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:686: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Pipeline(steps=[('preprocessor',\n",
              "                 ColumnTransformer(transformers=[('cat',\n",
              "                                                  OneHotEncoder(handle_unknown='ignore',\n",
              "                                                                sparse=False),\n",
              "                                                  ['Card Type', 'Entry Mode',\n",
              "                                                   'Transaction Type',\n",
              "                                                   'Merchant Group',\n",
              "                                                   'Transaction Country',\n",
              "                                                   'Issuing Bank']),\n",
              "                                                 ('non_cat',\n",
              "                                                  Pipeline(steps=[('imputer',\n",
              "                                                                   SimpleImputer(strategy='median')),\n",
              "                                                                  ('scaler',\n",
              "                                                                   StandardScaler())]),\n",
              "                                                  Index(['Transaction ID', 'Amount', 'Age'], dtype='object'))])),\n",
              "                ('classifier',\n",
              "                 MLPClassifier(hidden_layer_sizes=(64,), max_iter=10,\n",
              "                               random_state=42))])"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,\n",
              "                 ColumnTransformer(transformers=[(&#x27;cat&#x27;,\n",
              "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,\n",
              "                                                                sparse=False),\n",
              "                                                  [&#x27;Card Type&#x27;, &#x27;Entry Mode&#x27;,\n",
              "                                                   &#x27;Transaction Type&#x27;,\n",
              "                                                   &#x27;Merchant Group&#x27;,\n",
              "                                                   &#x27;Transaction Country&#x27;,\n",
              "                                                   &#x27;Issuing Bank&#x27;]),\n",
              "                                                 (&#x27;non_cat&#x27;,\n",
              "                                                  Pipeline(steps=[(&#x27;imputer&#x27;,\n",
              "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
              "                                                                  (&#x27;scaler&#x27;,\n",
              "                                                                   StandardScaler())]),\n",
              "                                                  Index([&#x27;Transaction ID&#x27;, &#x27;Amount&#x27;, &#x27;Age&#x27;], dtype=&#x27;object&#x27;))])),\n",
              "                (&#x27;classifier&#x27;,\n",
              "                 MLPClassifier(hidden_layer_sizes=(64,), max_iter=10,\n",
              "                               random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,\n",
              "                 ColumnTransformer(transformers=[(&#x27;cat&#x27;,\n",
              "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,\n",
              "                                                                sparse=False),\n",
              "                                                  [&#x27;Card Type&#x27;, &#x27;Entry Mode&#x27;,\n",
              "                                                   &#x27;Transaction Type&#x27;,\n",
              "                                                   &#x27;Merchant Group&#x27;,\n",
              "                                                   &#x27;Transaction Country&#x27;,\n",
              "                                                   &#x27;Issuing Bank&#x27;]),\n",
              "                                                 (&#x27;non_cat&#x27;,\n",
              "                                                  Pipeline(steps=[(&#x27;imputer&#x27;,\n",
              "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
              "                                                                  (&#x27;scaler&#x27;,\n",
              "                                                                   StandardScaler())]),\n",
              "                                                  Index([&#x27;Transaction ID&#x27;, &#x27;Amount&#x27;, &#x27;Age&#x27;], dtype=&#x27;object&#x27;))])),\n",
              "                (&#x27;classifier&#x27;,\n",
              "                 MLPClassifier(hidden_layer_sizes=(64,), max_iter=10,\n",
              "                               random_state=42))])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocessor: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat&#x27;,\n",
              "                                 OneHotEncoder(handle_unknown=&#x27;ignore&#x27;,\n",
              "                                               sparse=False),\n",
              "                                 [&#x27;Card Type&#x27;, &#x27;Entry Mode&#x27;, &#x27;Transaction Type&#x27;,\n",
              "                                  &#x27;Merchant Group&#x27;, &#x27;Transaction Country&#x27;,\n",
              "                                  &#x27;Issuing Bank&#x27;]),\n",
              "                                (&#x27;non_cat&#x27;,\n",
              "                                 Pipeline(steps=[(&#x27;imputer&#x27;,\n",
              "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
              "                                                 (&#x27;scaler&#x27;, StandardScaler())]),\n",
              "                                 Index([&#x27;Transaction ID&#x27;, &#x27;Amount&#x27;, &#x27;Age&#x27;], dtype=&#x27;object&#x27;))])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat</label><div class=\"sk-toggleable__content\"><pre>[&#x27;Card Type&#x27;, &#x27;Entry Mode&#x27;, &#x27;Transaction Type&#x27;, &#x27;Merchant Group&#x27;, &#x27;Transaction Country&#x27;, &#x27;Issuing Bank&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;, sparse=False)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">non_cat</label><div class=\"sk-toggleable__content\"><pre>Index([&#x27;Transaction ID&#x27;, &#x27;Amount&#x27;, &#x27;Age&#x27;], dtype=&#x27;object&#x27;)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MLPClassifier</label><div class=\"sk-toggleable__content\"><pre>MLPClassifier(hidden_layer_sizes=(64,), max_iter=10, random_state=42)</pre></div></div></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Este comando clf.fit(X_train, y_train) é utilizado para treinar o modelo de aprendizado de máquina no conjunto de treinamento fornecido."
      ],
      "metadata": {
        "id": "-MI_0mBvNQwO"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Avaliação do modelo no conjunto de teste**"
      ],
      "metadata": {
        "id": "3UBNaHCVNQzA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = clf.predict(X_test)\n",
        "y_pred_proba = clf.predict_proba(X_test)[:, 1]"
      ],
      "metadata": {
        "id": "cPVagIByas0G"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "O comando y_pred = clf.predict(X_test) é utilizado para fazer previsões das classes para as amostras no conjunto de teste (X_test) usando o modelo treinado (clf). A variável resultante y_pred contém as previsões de classe para cada amostra no conjunto de teste."
      ],
      "metadata": {
        "id": "8CpKqeHJNWzu"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Métricas de desempenho**"
      ],
      "metadata": {
        "id": "fQOLc7O4NbY9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Matriz de Confusão:\")\n",
        "print(confusion_matrix(y_test, y_pred))\n",
        "print(\"\\nRelatório de Classificação:\")\n",
        "print(classification_report(y_test, y_pred))\n",
        "print(\"\\nAUC-ROC:\", roc_auc_score(y_test, y_pred_proba))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0O3UusUrNb1O",
        "outputId": "226a877e-3c99-4cb8-93b4-425f9e8cfe4f"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Matriz de Confusão:\n",
            "[[142   0]\n",
            " [  8   0]]\n",
            "\n",
            "Relatório de Classificação:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "          No       0.95      1.00      0.97       142\n",
            "         Yes       0.00      0.00      0.00         8\n",
            "\n",
            "    accuracy                           0.95       150\n",
            "   macro avg       0.47      0.50      0.49       150\n",
            "weighted avg       0.90      0.95      0.92       150\n",
            "\n",
            "\n",
            "AUC-ROC: 0.3609154929577465\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, msg_start, len(result))\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, msg_start, len(result))\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, msg_start, len(result))\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Possui o objetivo de fornecer métricas de desempenho detalhadas para avaliar o modelo de classificação."
      ],
      "metadata": {
        "id": "uWnC-ecBN1YN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Mapeação 'Yes' para 1 e 'No' para 0**"
      ],
      "metadata": {
        "id": "3eAorKqUNeu-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_test_binary = y_test.map({'No': 0, 'Yes': 1})\n",
        "y_pred_proba = clf.predict_proba(X_test)[:, 1]"
      ],
      "metadata": {
        "id": "p7zXg9rGN2pI"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Possui o objetivo preparar o conjunto de teste para avaliação em um problema de classificação binária. O primeiro comando, y_test_binary = y_test.map({'No': 0, 'Yes': 1}), converte as classes do conjunto de teste de 'No' para 0 e 'Yes' para 1. Isso facilita a comparação direta entre as classes previstas e reais. O segundo comando, y_pred_proba = clf.predict_proba(X_test)[:, 1], obtém as probabilidades previstas para a classe positiva, útil para análises detalhadas da confiança do modelo. Ambos os passos são comuns na avaliação de modelos de classificação binária."
      ],
      "metadata": {
        "id": "Ne1d73F9PRDK"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Calcular a curva ROC**"
      ],
      "metadata": {
        "id": "aBdndPqLPRF6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)\n",
        "roc_auc = auc(fpr, tpr)"
      ],
      "metadata": {
        "id": "G-wOe96GN2wr"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Tem o objetivo de calcular as métricas relacionadas à curva ROC (Receiver Operating Characteristic) para avaliar o desempenho de um modelo de classificação binária."
      ],
      "metadata": {
        "id": "QA2gvL8BPL6_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Plotar a curva ROC**"
      ],
      "metadata": {
        "id": "3TyRRw-jPL_G"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(8, 8))\n",
        "plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = {:.2f})'.format(roc_auc))\n",
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
        "plt.xlim([0.0, 1.0])\n",
        "plt.ylim([0.0, 1.05])\n",
        "plt.xlabel('Taxa de Falsos Positivos (FPR)')\n",
        "plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')\n",
        "plt.title('Curva ROC')\n",
        "plt.legend(loc=\"lower right\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 718
        },
        "id": "wkErBeU-N20B",
        "outputId": "bcab0c08-f000-4015-c50e-17467c333531"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x800 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAr4AAAK9CAYAAADCE2/bAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACS8ElEQVR4nOzdd3hU1cLF4TUJaaQBQqiRIiAgSO8gIgmBIFU6gqCgFBvIVbFQ5IroFcSCNKWD9Kq0EKkCoiAKlya9SGiBhPQy5/uDj7lGWjJMcpLM732ePDJ7zsysTAxZ7Oyzj8UwDEMAAABALudidgAAAAAgK1B8AQAA4BQovgAAAHAKFF8AAAA4BYovAAAAnALFFwAAAE6B4gsAAACnQPEFAACAU6D4AgAAwClQfAEAAOAUKL4AcA/Hjx/XSy+9pDJlysjT01N+fn5q2LChPv/8c8XHx5sdL8M2b94si8Vi+3B1dVVAQIA6duyoQ4cO3fVx33//vVq0aKGHHnpInp6eKl++vIYOHaqrV6/e87U6dOigIkWKyN3dXQEBAWrdurWWLVuWGZ8aANyXxTAMw+wQAJAd/fDDD+rUqZM8PDzUq1cvVa5cWUlJSdq+fbuWLl2q3r17a+rUqWbHzJDNmzeradOmevXVV1W7dm0lJyfrjz/+0OTJk+Xt7a0DBw6oSJEiaR4zdOhQjRs3TlWrVlX37t1VoEAB7d27V9OnT1fBggUVHh6uRx99NM1jRowYoQ8++EDlypVTt27dVLJkSV29elVr1qzR5s2bNW/ePHXv3j0rP3UAkAwAwG1OnDhh+Pj4GBUqVDD++uuv2+7/888/jQkTJjjktWJiYhzyPOmxadMmQ5KxePHiNOOTJk0yJBkff/xxmvH58+cbkowuXboYKSkpae77+eefjbx58xpVqlQxkpOTbeOLFy82JBkdO3Y0kpKSbsuwbt06Y/Xq1Q78rAAgfVjqAAB38MknnygmJkbffvutihYtetv9ZcuW1WuvvSZJOnXqlCwWi2bOnHnbcRaLRSNHjrTdHjlypCwWiw4ePKju3bsrf/78atSokT799FNZLBadPn36tucYNmyY3N3dde3aNUnStm3b1KlTJz388MPy8PBQYGCgBg8e/EBLLxo3bizp5tKOvxs1apTy58+vqVOnytXVNc19derU0VtvvaX9+/dryZIltvH3339fBQoU0PTp0+Xm5nbba4WEhOjpp5+2OysA2IviCwB3sHr1apUpU0YNGjTIlOfv1KmT4uLiNGbMGPXr10+dO3eWxWLRokWLbjt20aJFat68ufLnzy9JWrx4seLi4jRgwAB9+eWXCgkJ0ZdffqlevXrZnefUqVOSZHsNSfrzzz915MgRtW3bVn5+fnd83K3X/P77722POXz4sNq1aydfX1+78wBAZshjdgAAyG6io6N1/vx5tW3bNtNeo2rVqpo/f36asXr16mnhwoX617/+ZRv75ZdfdOLEiTSzxh9//LG8vLxst1988UWVLVtW77zzjs6cOaOHH374vq9/48YNXblyxbbG9/XXX5fFYtEzzzxjO+bgwYO2rHdTqlQp+fn52U6Mu/XfKlWq3DcDAGQ1ZnwB4B+io6MlKVNnLPv373/bWJcuXbRnz540yw0WLlwoDw+PNCX876U3NjZWV65cUYMGDWQYhn777bd0vf7zzz+vQoUKqVixYmrRooWioqI0Z84c1a5d23bMjRs3JN3/ffD19bW9Z1nx3gGAvSi+APAPt36tf6v4ZYbSpUvfNtapUye5uLho4cKFkiTDMLR48WK1bNkyzVKDM2fOqHfv3ipQoIB8fHxUqFAhNWnSRJIUFRWVrtcfPny4wsLCtHz5cvXq1UtRUVFycUn7I+FWeb3f+3Djxg3bsVnx3gGAvVjqAAD/4Ofnp2LFiunAgQPpOt5isdxxPDU19a6P+fus7S3FihVT48aNtWjRIr3zzjvatWuXzpw5o48//jjNcwYHBysyMlJvvfWWKlSoIG9vb50/f169e/eW1WpNV+YqVaooKChIktSuXTvFxcWpX79+atSokQIDAyVJFStWlCT98ccfd32e06dPKzo6WpUqVZIkVahQQZK0f//+dOUAgKzEjC8A3MHTTz+t48ePa+fOnfc99tYJYdevX08zfqcdGu6nS5cu+v3333XkyBEtXLhQefPmVevWrW3379+/X0ePHtW4ceP01ltvqW3btgoKClKxYsUy/Fp/N3bsWCUkJOjDDz+0jZUvX17ly5fXihUr7jqDO3v2bEmy7dJQvnx5Pfroo1q5cqViYmIeKBMAOBrFFwDu4M0335S3t7f69u2rixcv3nb/8ePH9fnnn0u6OUNcsGBBbd26Nc0xX3/9dYZf95lnnpGrq6u+++47LV68WE8//bS8vb1t99/aUsz427WHDMOwZbHXI488omeeeUYzZ85URESEbXz48OG6du2a+vfvf9sM9p49e/Txxx+rcuXKaU6KGzVqlK5evaq+ffsqJSXlttfasGGDbRcIAMhKLHUAgDt45JFHNH/+fHXp0kUVK1ZMc+W2HTt2aPHixerdu7ft+L59+2rs2LHq27evatWqpa1bt+ro0aMZft2AgAA1bdpU48eP140bN9SlS5c091eoUEGPPPKIhg4dqvPnz8vPz09Lly617fH7IP71r39p0aJFmjBhgsaOHStJ6tGjh3755Rd9/vnnOnjwoHr06KH8+fPbrtz20EMPacmSJWn26+3SpYv279+vDz/8UL/99luaK7etW7dO4eHht+1oAQBZwtzrZwBA9nb06FGjX79+RqlSpQx3d3fD19fXaNiwofHll18aCQkJtuPi4uKMF154wfD39zd8fX2Nzp07G5cuXTIkGSNGjLAdN2LECEOScfny5bu+5rRp0wxJhq+vrxEfH3/b/QcPHjSCgoIMHx8fo2DBgka/fv2M33//3ZBkzJgx456fz92u3HbLk08+afj5+RnXr19PM75ixQojODjYyJ8/v+Hh4WGULVvWeOONN+75eYSHhxtt27Y1AgICjDx58hiFChUyWrdubaxcufKeGQEgs1gM42+/LwMAAAByKdb4AgAAwClQfAEAAOAUKL4AAABwChRfAAAAOAWKLwAAAJwCxRcAAABOwekuYGG1WvXXX3/J19dXFovF7DgAAAD4B8MwdOPGDRUrVkwuLo6bp3W64vvXX38pMDDQ7BgAAAC4j7Nnz6pEiRIOez6nK76+vr6Sbr6Rfn5+JqcBAADAP0VHRyswMNDW2xzF6YrvreUNfn5+FF8AAIBszNHLUjm5DQAAAE6B4gsAAACnQPEFAACAU6D4AgAAwClQfAEAAOAUKL4AAABwChRfAAAAOAWKLwAAAJwCxRcAAABOgeILAAAAp0DxBQAAgFOg+AIAAMApUHwBAADgFCi+AAAAcAoUXwAAADgFii8AAACcAsUXAAAAToHiCwAAAKdA8QUAAIBToPgCAADAKVB8AQAA4BQovgAAAHAKphbfrVu3qnXr1ipWrJgsFotWrFhx38ds3rxZNWrUkIeHh8qWLauZM2dmek4AAADkfKYW39jYWFWtWlUTJ05M1/EnT55Uq1at1LRpU+3bt0+vv/66+vbtq/Xr12dyUgAAAOR0ecx88ZYtW6ply5bpPn7y5MkqXbq0xo0bJ0mqWLGitm/frs8++0whISGZFRMAAAC5gKnFN6N27typoKCgNGMhISF6/fXX7/qYxMREJSYm2m5HR0dnVjwAAICc68hiacdwKemG2UmkeGumPG2OKr4REREqXLhwmrHChQsrOjpa8fHx8vLyuu0xH330kUaNGpVVEQEAAHKmHcOlyMNmp7gpIXOeNkcVX3sMGzZMQ4YMsd2Ojo5WYGCgiYkAAACyoVszvRYXybtolr98SqpFeVyNmzdcrZIuOPw1clTxLVKkiC5evJhm7OLFi/Lz87vjbK8keXh4yMPDIyviAQAA5HzeRaWXzmXpS86fv19jx25XeHgvFSrkLUVHS4P9Hf46OWof3/r16ys8PDzNWFhYmOrXr29SIgAAADyIqVP36Nlnl2n//ksKCZmrGzcS7/8gO5lafGNiYrRv3z7t27dP0s3tyvbt26czZ85IurlMoVevXrbj+/fvrxMnTujNN9/U4cOH9fXXX2vRokUaPHiwGfEBAADwAMaN26GXXvpexv+vcKhTp7i8vd0z7fVMLb6//vqrqlevrurVq0uShgwZourVq2v48OGSpAsXLthKsCSVLl1aP/zwg8LCwlS1alWNGzdO33zzDVuZAQAA5CCGYWjEiE0aOjTMNjZ0aH1NmtRKLi6WTHtdi2Hc6tjOITo6Wv7+/oqKipKfn5/ZcQAAALKHKSWkmPOST/FMXeNrGIbeeGODPvtsl21s9OimevfdxrJYbpbezOprOerkNgAAAORcqalW9e//vb755jfb2Gefhej11+tlyetTfAEAAJDpUlKs6tlzuRYsOCBJslikadNa64UXamRZBoovAAAAMp2rq0Xe3m6SpDx5XDR3bnt16VI5SzNQfAEAAJDpLBaLpkx5Wqmphp55pqKefrp8lmeg+AIAACBLuLq6aMaMtqa9fo66gAUAAAByhkuXYvXUU7P0xx8X739wFqH4AgAAwKHOno1S48YztGnTKQUHz9HRo1fNjiSJpQ4AAABwoGPHIhUUNFunT0dJkjw8XE1O9D8UXwAAADjEgQOXFBw8RxERMZKksmULaOPGnipZMp+5wf4fxRcAAAAP7Ndf/1JIyFxFRsZLkipXDlBYWE8VKeJjcrL/YY0vAAAAHsjWraf11FOzbKW3du1i2rKld7YqvRLFFwAAAA9g3bpjCgmZqxs3kiRJTzxRUhs39lKBAl4mJ7sdxRcAAAB2O348UgkJKZKkFi3Kau3aHvLz8zA51Z2xxhcAAAB2GzSojqKjE7VnzwXNn/+M3N2zzy4O/0TxBQAAwAMZNqyxrFZDLi4Ws6PcE0sdAAAAkG4ff7xdy5Ydum08u5deiRlfAAAApINhGHr33R/10Ufb5ebmotWruykkpKzZsTKEGV8AAADck9Vq6NVX1+qjj7ZLkpKTrTpw4JLJqTKOGV8AAADcVUqKVX37rtKsWb/bxiZODNXAgbVNTGUfii8AAADuKCkpVd27L9XSpTfX9Lq4WDRjRlv16lXV5GT2ofgCAADgNnFxyXrmmUVat+6YJMnNzUULFnRUhw4VTU5mP4ovAAAA0oiOTlTr1t9p69bTkiQvrzxavrxLjjuZ7Z8ovgAAAEjjwIFL+vnnc5IkPz8Pff99NzVuXNLkVA+OXR0AAACQRoMGgVq8uJOKFPHRjz/2yhWlV2LGFwAAAHfQuvWjOnastLy93c2O4jDM+AIAADi5I0eu6PPwx24bz02lV2LGFwAAwKn9/nuEmjefq0uXGsiaGK3BoWfNjpRpmPEFAABwUrt2ndOTT87SpUuxkqTZv1ZVYnLurYe59zMDAADAXf3440kFBc3W9esJkqT6ZS7qx/6z5OFmNTlZ5mGpAwAAgJP5/vuj6thxkRITUyVJTz1VWivbzJBPSoLJyTIXM74AAABOZOHCA2rffqGt9LZuXV4//NBdPp4pJifLfBRfAAAAJ/HNN3vVrdtSpaTcXM7QtWtlLV3aWZ6ezrEIgOILAADgBG7cSNSIEZtlGDdv9+1bXXPntpebm6u5wbIQxRcAAMAJ+Pp6aMOGZ/XQQ14aPLiepk5tLVdX56qCzjGvDQAAAD32WIB+/72/ihXzlcViMTtOlnOumg8AAOAkUlOtmjp1j2097y3Fi/s5ZemVKL4AAAC5TnJyqnr1WqGXXvpeL7ywSlarYXakbIHiCwAAkIskJKSoU6fFmj9/vyRp3rw/tGfPXyanyh5Y4wsAAJBLxMYmqV27hdq48YQkycPDVYsXd1Lt2sVNTpY9UHwBAABygevXE9Sq1Xzt2HFWkuTt7aaVK7uqWbMyJifLPii+AAAAOdzly7Fq3nyu9u2LkCT5+3to7doeql8/0ORk2QvFFwAAIAc7fz5aQUFzdPjwFUlSoUJ5tWFDT1WrVsTkZNkPxRcAACAHGzJkg630lijhp7CwnqpQoaDJqbInii8AAEAONmlSKx08eFlxcckKD++lUqXymR0p26L4AgAA5GAFCngpLKynrFZDxYr5mh0nW2MfXwAAgBzk55/PKTIyPs1YkSI+lN50oPgCAADkEBs2HFfTprMUGjpPN24kmh0nx6H4AgAA5ADLlx9S69bfKT4+RT//fF5jx243O1KOQ/EFAADI5ubO/UOdOi1WUlKqJKl9+woaPryJyalyHoovAABANjZ58q/q1Wu5UlMNSVLPno9r0aJO8vBgj4KMovgCAABkU//5z08aMOAHGTc7rwYOrKWZM9spTx4qnD141wAAALIZwzD0/vs/6s03N9rG3nqrob76KlQuLhYTk+VszJEDAABkM/Pn79e//73NdnvMmKc0bFhjExPlDsz4AgAAZDNdulRWu3YVJElffNGC0usgzPgCAABkM3nyuGjBgmf0448n1bJlObPj5BrM+AIAAJgsPj5Zx49Hphnz8MhD6XUwii8AAICJbtxIVGjofD3xxEydOHHN7Di5GsUXAADAJJGR8QoKmqPNm0/pr79uqEOHhbJaDbNj5Vqs8QUAADBBRESMmjefo/37L0mSChTw0rRprdmuLBNRfAEAALLYmTNRCgqarT//vLmut3Bhb23c2EuVKweYnCx3o/gCAABkoT//vKpmzWbr7NloSdLDD/tr48aeKlfuIZOT5X4UXwAAgCzyxx8X1bz5HF28GCtJKleugDZu7KWHH/Y3OZlzoPgCAABkgYiIGD355Exdu5YgSapSJUBhYT1VuLCPycmcB7s6AAAAZIEiRXz0yit1JEl16xbX5s29Kb1ZjBlfAACALDJy5JMqWtRXPXpUka+vh9lxnA7FFwAAIJNc2vGdAo58ICXdkCRZJPW3SJpvaqw7i71gdoJMR/EFAADIBDNm/KZXBvxXq/skqGnZ82bHST93X7MTZBqKLwAAgIN98cXPeu21dZLc1Hp6d/02ZKrKlcoBSxvcfaWGo81OkWkovgAAAA5iGIbGjNmm997bZBt7oc5ePfKwh/TSOROTQWJXBwAAAIcwDEPDhoWnKb3vh+7VhLbr5ELjyhaY8QUAAHhAVquhl19eo0mTfrWNffJJkP7lN02KMTEY0uDfHwAAAA8gJcWq3r1X2EqvxSJNmtRK//pXQ5OT4Z+Y8QUAAHgAPXsu14IFByRJrq4WzZzZTs8++7jJqXAnzPgCAAA8gI4dK8rFxSJ3d1ctWdKZ0puNMeMLAADwAJ55ppJmzmyrIkV8FBz8iNlxcA8UXwAAgAxISEiRp2faCtWzZ1WT0iAjWOoAAACQTn/9dUM1a07V11//YnYU2IHiCwAAkA4nT15T48YzdPDgZQ0atMZ2QhtyDpY6AAAA3Mfhw1cUFDRb58/fkCSVLp1PdeoUNzkVMoriCwAAcA+//XZBzZvP1ZUrcZKkihULKiysp4oX9zM5GTKK4gsAAHAXO3acVWjoPEVFJUqSqlcvovXrn1WhQt4mJ4M9WOMLAABwBxs3nlBw8Bxb6W3QIFA//vgcpTcHo/gCAAD8ww8/HFWrVvMVF5csSQoKKqMNG55VvnyeJifDg6D4AgAA/EOJEn7Km9dNktS27aNavbqbvL3dTU6FB0XxBQAA+IeqVYto7doe6tevhhYv7nTbBSuQM/FVBAAAkGQYhiwWi+12vXolVK9eCRMTwdGY8QUAAE7NMAyNHLlZL764WoZhmB0HmYgZXwAA4LQMw9Abb2zQZ5/tkiT5+npo/PgQk1Mhs1B8AQCAU0pNtap//+/1zTe/2cYeftjfxETIbBRfAADgdJKTU9Wz53ItXPhfSZLFIk2b1lovvFDD5GTITBRfAADgVBISUtSp02J9//1RSVKePC6aO7e9unSpbHIyZDaKLwAAcBoxMUlq23aBfvzxpCTJw8NVS5d2VqtW5U1OhqxA8QUAAE7h2rV4hYbO165d5yRJPj7uWrWqq5o2LW1yMmQVii8AAHAKKSlWXb+eIEnKl89T69b1UN267NPrTNjHFwAAOIVChbwVFtZT9eqV0JYtvSm9TogZXwAA4DRKlPDTjh3Pp7lCG5wHM74AACBXOnDgkjp1WqzY2KQ045Re58WMLwAAyHV+/fUvhYTMVWRkvKKjE7VqVVd5eFB7nB0zvgAAIFfZuvW0nnpqliIj4yXd3M0hLi7Z5FTIDii+AAAg11i37phatJirGzduLm944omS2rixl/Ln9zI5GbIDii8AAMgVli49qDZtvlN8fIokqUWLslq7tof8/DxMTobsguILAAByvFmz9qlz5yVKTrZKkjp2rKSVK7sqb143k5MhO6H4AgCAHG3ixN3q3XulrFZDktS7dzV9990zcnd3NTkZshuKLwAAyLGsVkNr1hyz3X7llTr69ts2ypOHioPbsa8HAADIsVxcLFqypJNCQ+erQYMS+ve/n2KfXtwVxRcAAORoXl5uWr/+WZY24L5M/z3AxIkTVapUKXl6eqpu3bravXv3PY+fMGGCHn30UXl5eSkwMFCDBw9WQkJCFqUFAABmSkmx6q23wnTmTFSacUov0sPU4rtw4UINGTJEI0aM0N69e1W1alWFhITo0qVLdzx+/vz5evvttzVixAgdOnRI3377rRYuXKh33nkni5MDAICslpSUqm7dluqTT3YoKGi2Ll6MMTsSchhTi+/48ePVr18/9enTR5UqVdLkyZOVN29eTZ8+/Y7H79ixQw0bNlT37t1VqlQpNW/eXN26dbvvLDEAAMjZ4uKS1bbtAi1ZclCSdOrUde3bF2FyKuQ0phXfpKQk7dmzR0FBQf8L4+KioKAg7dy5846PadCggfbs2WMruidOnNCaNWsUGhp619dJTExUdHR0mg8AAJBzREcnqkWLuVq37ubuDV5eebR6dTeFhJQ1ORlyGtNObrty5YpSU1NVuHDhNOOFCxfW4cOH7/iY7t2768qVK2rUqJEMw1BKSor69+9/z6UOH330kUaNGuXQ7AAAIGtcvRqnFi3m6ddf/5Ik+fl56Pvvu6lx45ImJ0NOZPrJbRmxefNmjRkzRl9//bX27t2rZcuW6YcfftDo0aPv+phhw4YpKirK9nH27NksTAwAAOx14cINNWky01Z6H3rISz/+2IvSC7uZNuNbsGBBubq66uLFi2nGL168qCJFitzxMe+//7569uypvn37SpKqVKmi2NhYvfjii3r33Xfl4nJ7j/fw8JCHB9foBgAgJzl16rqCgmbr+PFrkqSiRX0UFtZTjz0WYHIy5GSmzfi6u7urZs2aCg8Pt41ZrVaFh4erfv36d3xMXFzcbeXW1fXm9iWGYWReWAAAkKW++26/rfSWLOmvbdv6UHrxwEy9gMWQIUP03HPPqVatWqpTp44mTJig2NhY9enTR5LUq1cvFS9eXB999JEkqXXr1ho/fryqV6+uunXr6tixY3r//ffVunVrWwEGAAA539tvN9KZM1HatOmUNm7spRIl/MyOhFzA1OLbpUsXXb58WcOHD1dERISqVaumdevW2U54O3PmTJoZ3vfee08Wi0Xvvfeezp8/r0KFCql169b68MMPzfoUAABAJrBYLJo4sZWiohKUP7+X2XGQS1gMJ1sjEB0dLX9/f0VFRcnPj389AgCQHfz440m5u7uqUaOHzY7iWFNKSDHnJZ/i0kvnzE6TY2RWX8tRuzoAAIDc5/vvjyo0dJ5atZqvvXsvmB0HuRjFFwAAmGbhwgNq336hEhNTFR2dqC+++NnsSMjFKL4AAMAU3367V926LVVKilWS1LVrZU2b1trkVMjNKL4AACDLTZiwS337rtatM4369q2uuXPby82NXZqQeSi+AAAgyxiGodGjt2jw4PW2sSFD6mnq1NZydaWWIHOZup0ZAABwHoZh6M03w/TppzttYyNHNtHw4U1ksVhMTAZnQfEFAABZ4pdf/tK4cf8rvePGNdeQIXe+WiuQGfidAgAAyBJ16hTX5MlPy8XFoilTnqb0Issx4wsAALLMiy/WVJMmJfXoowXNjgInxIwvAADIFLGxSVqz5s/bxim9MAvFFwAAONz16wlq3nyunn56vhYuPGB2HEASxRcAADjY5cuxatp0lnbsOCvDkF55Za1u3Eg0OxbAGl8AAOA4585FKzh4jg4fviJJKlQorzZs6ClfXw+TkwEUXwAA4CAnTlxTs2azderUdUlS8eK+2rixlypUYE0vsgeKLwAAeGAHD15WUNBsXbgQI0kqUya/wsN7qVSpfOYGA/6G4gsAAB7I3r0X1Lz5HF29Gi9JqlSpkMLCeqpYMV+TkwFpUXwBAIDdEhNT1K7dAlvprVmzqNate1YFC+Y1ORlwO3Z1AAAAdvPwyKO5czvIyyuPGjV6WOHhvSi9yLaY8QUAAA/kiSdK6scfn1OVKgHy9nY3Ow5wV8z4AgCADNm9+7wMw0gzVq9eCUovsj2KLwAASLfJk39VvXrf6L33fjQ7CpBhFF8AAJAu//nPTxow4AcZhjRmzHatX3/M7EhAhrDGFwAA3JNhGBo+fJP+/e9ttrG33mqo5s0f+d9BRxZLO4ZLSTdMSJiNxV4wOwH+huILAADuymo1NHjwOn3xxW7b2IcfPqV33mmc9sAdw6XIw1mcLgdxZ0/j7IDiCwAA7ig11ap+/VZrxox9trEvvmihV16pe/vBt2Z6LS6Sd9GsCZhTuPtKDUebnQKi+AIAgDtISkrVs88u0+LFByVJLi4WffttG/XuXe3eD/QuKr10LvMDAnag+AIAgNsMHrzOVnrd3Fw0f/4z6tixksmpgAfDrg4AAOA2b77ZUCVK+MnTM49WruxK6UWuwIwvAAC4TcmS+bRxY09dvBirJ54oaXYcwCEovgAAQJcuxcrPz0Oenv+rBo8+WlCPPlrQxFSAY7HUAQAAJ3fmTJQaNZquTp0WKzk51ew4QKah+AIA4MT+/POqGjWarj//jNT33x/Vm2+GmR0JyDQsdQAAwEn98cdFNW8+RxcvxkqSypUroMGD65ucCsg8FF8AAJzQ7t3n1aLFXF27liBJqlIlQGFhPVW4sI/JyYDMk+Hia7VatWXLFm3btk2nT59WXFycChUqpOrVqysoKEiBgYGZkRMAADjI5s2n1Lr1d4qJSZIk1a1bXGvW9FCBAl4mJwMyV7rX+MbHx+vf//63AgMDFRoaqrVr1+r69etydXXVsWPHNGLECJUuXVqhoaHatWtXZmYGAAB2WrPmT7VsOc9Weps2LaWwsJ6UXjiFdM/4li9fXvXr19e0adMUHBwsNze32445ffq05s+fr65du+rdd99Vv379HBoWAADYb9Omk2rXboGSk62SpFatymnx4k7y8rr9ZzqQG1kMwzDSc+ChQ4dUsWLFdD1pcnKyzpw5o0ceeeSBwmWG6Oho+fv7KyoqSn5+fmbHAQAgy8TGJql587naseOsOnd+THPmtJe7u6tjnnxKCSnmvORTXHrpnGOeE04rs/paumd801t6JcnNzS1bll4AAJyZt7e7fvihu776areGDWskV1d2NYVzcej/8cuWLdPjjz/uyKcEAAB2MgxDN24kphnLl89T7733BKUXTinD/9dPmTJFHTt2VPfu3fXzzz9Lkn788UdVr15dPXv2VMOGDR0eEgAAZIxhGBo2LFz16n2rK1fizI4DZAsZKr5jx47VK6+8olOnTmnVqlV66qmnNGbMGPXo0UNdunTRuXPnNGnSpMzKCgAA0sFqNTRo0Bp9/PFPOnjwslq0mMuliAFlcB/fGTNmaNq0aXruuee0bds2NWnSRDt27NCxY8fk7e2dWRkBAEA6paRY9fzzKzVnzh+SJItF6tevhtzcHHQSG5CDZaj4njlzRk899ZQkqXHjxnJzc9OoUaMovQAAZAOJiSnq1m2pli8/LElydbVo1qx26tGD828AKYPFNzExUZ6enrbb7u7uKlCggMNDAQCAjImNTVKHDou0YcNxSZK7u6sWLeqotm0rmJwMyD4yfMni999/X3nz5pUkJSUl6d///rf8/f3THDN+/HjHpAMAAPcVFZWgp5/+Ttu3n5Ek5c3rphUruig4mK1Fgb/LUPF94okndOTIEdvtBg0a6MSJE2mOsVgsjkkGAADuKzo6Uc2azdaePRckSX5+HlqzprsaNnzY5GRA9pOh4rt58+ZMigEAAOzh4+OuqlULa8+eCypYMK/Wr39WNWoUNTsWkC1leKlDdHS0fv75ZyUlJalOnToqVKhQZuQCAADp4OJi0dSprZU3r5sGDKitSpX4uQzcTYaK7759+xQaGqqIiAhJkq+vrxYtWqSQkJBMCQcAAG6XmmpNc+U1V1cXffllqImJgJwhQxeweOutt1S6dGn99NNP2rNnj5o1a6aXX345s7IBAIB/+O23C3rssa+1f/9Fs6MAOU6GZnz37NmjDRs2qEaNGpKk6dOnq0CBAoqOjpafn1+mBAQAADft2HFWoaHzFBWVqODgOdq+/XmVLcu2okB6ZWjGNzIyUiVKlLDdzpcvn7y9vXX16lWHBwMAAP+zceMJBQfPUVRUoiTpkUcKqGDBvCanAnKWDJ/cdvDgQdsaX0kyDEOHDh3SjRs3bGOPP84VYgAAcJRVq46oU6fFSkpKlSQFBZXRihVd5O3tbnIyIGfJcPFt1qyZDMNIM/b000/LYrHIMAxZLBalpqY6LCAAAM7su+/2q2fP5UpNvfmzt23bR7VgQUd5emb4Rzjg9DL0XXPy5MnMygEAAP5h6tQ96t//e92ab+revYpmzmwrNzdXc4MBOVSGiu+sWbM0dOhQ2yWLAQBA5hg/fqfeeGOD7fZLL9XU11+3kosLV0gF7JWhk9tGjRqlmJiYzMoCAAD+n5vb/35EDx1aX5MmUXqBB5WhGd9/ru0FAACZ45VX6urGjSSlplr13ntPyGKh9AIPKsMr4/nGAwAga7zzTmOzIwC5SoaLb/ny5e9bfiMjI+0OBACAs0lOTtXzz69Shw4V1L59RbPjALlWhovvqFGj5O/vnxlZAABwOgkJKercebFWrz6qRYv+q9Wru6l580fMjgXkShkuvl27dlVAQEBmZAEAwKnExCSpbdsF+vHHm9uFWiw3Z38BZI4MFV/W9wIA4BjXrsUrNHS+du06J0ny8XHXqlVd1bRpaZOTAbkXuzoAAJDFLl2KVfPmc/T77xclSfnyeWrduh6qW7eEycmA3C1DxddqtWZWDgAAnMLZs1EKCpqjo0evSpICArwVFtZTjz9e2ORkQO6X7gtY9O/fX+fOnUvXsQsXLtS8efPsDgUAQG507FikGjeeYSu9gYF+2ratD6UXyCLpnvEtVKiQHnvsMTVs2FCtW7dWrVq1VKxYMXl6euratWs6ePCgtm/frgULFqhYsWKaOnVqZuYGACDHOXMmShcu3LwCatmyBbRxY0+VLJnP3FCAE7EYGVi4e/HiRX3zzTdasGCBDh48mOY+X19fBQUFqW/fvmrRooXDgzpKdHS0/P39FRUVJT8/P7PjAACczKpVRzRixGatWdNdRYv6mh3HcaaUkGLOSz7FpZfS9xti4G4yq69lqPj+3bVr13TmzBnFx8erYMGCeuSRR3LErg8UXwCA2VJTrXJ1Tfdqw5yB4gsHyqy+luF9fG/Jnz+/8ufP77AgAADkNuvWHdNvv13QsGFpLz2c60ovkEPYXXwBAMDdLV16UN26LVVyslWennk0eHB9syMBTo9/cgIA4GCzZu1T585LlJx8cxvQHTvOsRc+kA1QfAEAcKCJE3erd++VslpvFt3nnquq7757JkecBwPkdhRfAAAcZOzY7Xr55bW22y+/XFvTp7dVnjz8uAWyA7u+E+Pj4xUXF2e7ffr0aU2YMEEbNmxwWDAAAHIKwzD0zjvhGjYs3Db2zjuN9MUXLeXiwkwvkF3YVXzbtm2r2bNnS5KuX7+uunXraty4cWrbtq0mTZrk0IAAAGRnVquhV15Zq48+2m4b++ijZvrww2YsbwCyGbuK7969e9W48c2tWZYsWaLChQvr9OnTmj17tr744guHBgQAIDu7eDFGy5Ydst2eODFUb7/dyMREAO7GruIbFxcnX9+bV5vZsGGDOnToIBcXF9WrV0+nT592aEAAALKzokV9FRbWU4ULe2vWrHYaOLC22ZEA3IVdxbds2bJasWKFzp49q/Xr16t58+aSpEuXLnE1NACA03nssQD9+ecr6tWrqtlRANyDXcV3+PDhGjp0qEqVKqU6deqofv2bm3Jv2LBB1atXd2hAAACyk+joRH3wwRalpFjTjPv6epiUCEB62XXlto4dO6pRo0a6cOGCqlb9379umzVrpvbt2zssHAAA2cnVq3Fq0WKefv31L504cU3Tp7dl1wYgB7H7ksVFihRRkSJFdO7cOUlSiRIlVKdOHYcFAwAgO7lw4YaCg+fov/+9LEn6/vujOn36ukqXzm9yMgDpZddSB6vVqg8++ED+/v4qWbKkSpYsqXz58mn06NGyWq33fwIAAHKQU6euq3HjGbbSW7Soj7Zs6U3pBXIYu2Z83333XX377bcaO3asGjZsKEnavn27Ro4cqYSEBH344YcODQkAgFmOHLmioKA5OncuWpJUsqS/wsN76ZFHCpicDEBG2VV8Z82apW+++UZt2rSxjT3++OMqXry4Bg4cSPEFAOQKv/8eoebN5+rSpVhJ0qOPPqSNG3upRAl2MAJyIruKb2RkpCpUqHDbeIUKFRQZGfnAoQAAMNuuXefUsuU8Xb+eIEmqWrWwNmzoqYAAb5OTAbCXXWt8q1atqq+++uq28a+++irNLg8AAOREhmHonXfCbaW3fv0S2rTpOUovkMPZNeP7ySefqFWrVtq4caNtD9+dO3fq7NmzWrNmjUMDAgCQ1SwWixYv7qQnn5ylwoW9tWJFV/n4uJsdC8ADsqv4NmnSREeOHNHXX3+tw4cPS5I6dOiggQMHqlixYg4NCACAGR56KK/Cw3vJz89Dnp527/4JIBux+zu5ePHinMQGAMg1liw5qGbNSit/fi/bGEsbgNzFrjW+ZcuW1ciRI/Xnn386Og8AAFluwoRd6tRpsVq2nKcbNxLNjgMgk9hVfAcNGqQffvhBjz76qGrXrq3PP/9cERERjs4GAECmMgxDo0dv0eDB6yVJP/98XvPm7Tc5FYDMYlfxHTx4sH755RcdPnxYoaGhmjhxogIDA9W8eXPNnj3b0RkBAHA4wzD05pthGj58s21s5MgmeumlmuaFApCp7Cq+t5QvX16jRo3S0aNHtW3bNl2+fFl9+vRxVDYAADJFaqpVAwb8oE8/3WkbGzeuuUaMeFIWi8XEZAAy0wOfprp7927Nnz9fCxcuVHR0tDp16uSIXAAAZIrk5FT17r1S8+ffXNJgsUhTpjytfv3snOk9sljaMVxKuuHAlDlQ7AWzEwD3ZVfxPXr0qObNm6fvvvtOJ0+e1FNPPaWPP/5YHTp0kI+Pj6MzAgDgEAkJKeradYlWrjwiScqTx0WzZ7dTt25V7H/SHcOlyMMOSpgLuPuanQC4K7uKb4UKFVS7dm0NGjRIXbt2VeHChR2dCwAAh/vyy59tpdfDw1WLF3dS69aPPtiT3prptbhI3kUfMGEO5+4rNRxtdgrgruwqvkeOHFG5cuUcnQUAgEz1+uv19NNPZ7Vx4wmtXNlVzZqVcdyTexeVXjrnuOcD4HB2FV9KLwAgJ3Jzc9WCBR115MgVVa1axOw4ALJYuotvgQIFdPToURUsWFD58+e/51mvkZGRDgkHAMCDOH8+WjduJKlChYK2MU/PPJRewEmlu/h+9tln8vX1tf2Z7V4AANnZiRPX1KzZbCUnp2rbtj4qXTq/2ZEAmCzdxfe5556z/bl3796ZkQUAAIc4ePCygoJm68KFGEnSSy99rw0bepqcCoDZ7LqAhaurqy5dunTb+NWrV+Xq6vrAoQAAsNfevRf0xBMzbKW3UqVCmjmznbmhAGQLdhVfwzDuOJ6YmCh3d/cHCgQAgL22bz+jpk1n6erVeElSzZpFtWVLbxUrxt6yADK4q8MXX3whSbJYLPrmm2/SXKwiNTVVW7duVYUKFRybEACAdAgLO662bRcoPj5FktSo0cP6/vtu8vf3NDkZgOwiQ8X3s88+k3Rzxnfy5MlpljW4u7urVKlSmjx5smMTAgBwHytWHFaXLkuUlJQqSWre/BEtX95FefO6mZwMQHaSoaUOJ0+e1MmTJ9WkSRP9/vvvttsnT57UkSNHtH79etWtWzdDASZOnKhSpUrJ09NTdevW1e7du+95/PXr1zVo0CAVLVpUHh4eKl++vNasWZOh1wQA5B6HDl1Wx46LbKW3Q4eKWrWqK6UXwG3sWuO7adMm5c//4NvCLFy4UEOGDNGIESO0d+9eVa1aVSEhIXc8cU6SkpKSFBwcrFOnTmnJkiU6cuSIpk2bpuLFiz9wFgBAzlSxYiG9805jSVLPno9r4cKO8vCw6/pMAHI5i3G3M9X+YciQIRo9erS8vb01ZMiQex47fvz4dL143bp1Vbt2bX311VeSJKvVqsDAQL3yyit6++23bzt+8uTJ+s9//qPDhw/Lzc2+f8lHR0fL399fUVFR8vPzs+s5AADZi2EYWrbskNq3rygXlyzeZ35KCSnmvORTnEsWAw6SWX0t3f8k/u2335ScnGz7892k98IWSUlJ2rNnj4YNG2Ybc3FxUVBQkHbu3HnHx6xatUr169fXoEGDtHLlShUqVEjdu3fXW2+9dddt1BITE5WYmGi7HR0dna58AIDsyTAMHT16VY8++r+rsVksFj3zTCUTUwHICdJdfDdt2nTHP9vrypUrSk1NVeHChdOMFy5cWIcPH77jY06cOKEff/xRPXr00Jo1a3Ts2DENHDhQycnJGjFixB0f89FHH2nUqFEPnBcAYD6r1dCQIes1ZcoerV3bQ08+WcrsSAByELvW+P5TdHS0VqxYcdfC6ihWq1UBAQGaOnWqatasqS5duujdd9+9504Sw4YNU1RUlO3j7NmzmZoRAJA5UlOt6tdvlT7//GclJKSobdsFunw51uxYAHIQu1b/d+7cWU888YRefvllxcfHq1atWjp16pQMw9CCBQv0zDPP3Pc5ChYsKFdXV128eDHN+MWLF1WkSJE7PqZo0aJyc3NLs6yhYsWKioiIUFJS0h0vnuHh4SEPD48MfoYAgOwkKSlVPXsu16JF/5UkubhY9PnnLVSokLfJyQDkJHbN+G7dulWNG988g3b58uUyDEPXr1/XF198oX//+9/peg53d3fVrFlT4eHhtjGr1arw8HDVr1//jo9p2LChjh07JqvVahs7evSoihYtyhXjACCXio9PVocOC22l183NRQsXdlTv3tXMDQYgx7Gr+EZFRalAgQKSpHXr1umZZ55R3rx51apVK/3555/pfp4hQ4Zo2rRpmjVrlg4dOqQBAwYoNjZWffr0kST16tUrzclvAwYMUGRkpF577TUdPXpUP/zwg8aMGaNBgwbZ82kAALK5GzcSFRo6Xz/8cPNni6dnHq1c2VUdO3IiG4CMs2upQ2BgoHbu3KkCBQpo3bp1WrBggSTp2rVr8vRM/6Uhu3TposuXL2v48OGKiIhQtWrVtG7dOtsJb2fOnJGLy/+6eWBgoNavX6/Bgwfr8ccfV/HixfXaa6/prbfesufTAABkY5GR8WrZcp527z4vSfL1ddfq1d3UpEkpc4MByLHSvY/v33399dd67bXX5OPjo5IlS2rv3r1ycXHRl19+qWXLljlk14fMwj6+AJD9paZaVa/et/r1178kSfnze2r9+mdVu3Y2vGAR+/gCDpdZfc2upQ4DBw7Uzp07NX36dG3fvt02K1umTJl0r/EFAOBuXF1d9NZbDeXiYlHhwt7asqV39iy9AHIUu2Z8/+7Ww9N74QqzMeMLADnHggUHVLNmUZUr95DZUe6OGV/A4bLVjK8kzZ49W1WqVJGXl5e8vLz0+OOPa86cOQ4LBgBwLlevxt021rVr5exdegHkKHYV3/Hjx2vAgAEKDQ3VokWLtGjRIrVo0UL9+/fXZ5995uiMAIBcbvfu8ypf/it9/fUvZkcBkIvZtavDl19+qUmTJqlXr162sTZt2uixxx7TyJEjNXjwYIcFBADkbps3n1Lr1t8pJiZJgwatUWCgn1q3ftTsWAByIbtmfC9cuKAGDRrcNt6gQQNduHDhgUMBAJzDmjV/qmXLeYqJSZIkPflkKT35ZClzQwHItewqvmXLltWiRYtuG1+4cKHKlSv3wKEAALnf4sX/Vdu2C5SQkCJJCg0tpzVrusvXl8vMA8gcdi11GDVqlLp06aKtW7eqYcOGkqSffvpJ4eHhdyzEAAD83YwZv6lv39WyWm/uDNSpUyXNndtB7u6uJicDkJvZNeP7zDPPaPfu3SpYsKBWrFihFStWqGDBgtq9e7fat2/v6IwAgFzkiy9+1vPPr7KV3uefr6bvvnuG0gsg02V4xjc6Olo///yzkpKS9Nlnn6lQoUKZkQsAkAtNmLBLgwevt91+9dU6+uyzFnJxyRl7wQPI2TJUfPft26fQ0FBdvHhRhmHI19dXixYtUkhISGblAwDkIk2bllK+fJ66fj1B77//hEaNejLHXAAJQM6XoaUOb731lkqXLq3t27drz549atasmV5++eXMygYAyGWqVi2iNWu6a/z45vrgg6aUXgBZKkMzvnv27NGGDRtUo0YNSdL06dNVoEABRUdHc/lfAMBtUlKsslgkV9f/zbPUrx+o+vUDTUwFwFllaMY3MjJSJUqUsN3Oly+fvL29dfXqVYcHAwDkbImJKerUabEGDPhBhmGYHQcAMn5y28GDBxUREWG7bRiGDh06pBs3btjGHn/8ccekAwDkSLGxSWrffqHCwk5IkgoU8NLYsUEmpwLg7DJcfJs1a3bbv9yffvppWSwWGYYhi8Wi1NRUhwUEAOQsUVEJatVqvn766awkKW9eNzVrVtrkVACQweJ78uTJzMoBAMgFrlyJU0jIXO3de/Py9X5+HlqzprsaNnzY5GQAkMHiW7JkyczKAQDI4f7664aCg+fo4MHLkqSCBfNq/fpnVaNGUZOTAcBNdl2yGACAvzt58pqCguboxIlrkqRixXy1cWNPVazIRY4AZB8UXwDAAzl69KqeemqWzp+/eZJz6dL5tHFjL5Upk9/kZACQFsUXAPBA8uXzlI+PuySpYsWCCgvrqeLF2dsdQPaToX18AQD4p4AAb23c2Evt21fQli29Kb0Asi27Znzj4+NlGIby5s0rSTp9+rSWL1+uSpUqqXnz5g4NCADIfm5tX3lLiRJ+Wrasi4mJAOD+7Jrxbdu2rWbPni1Jun79uurWratx48apbdu2mjRpkkMDAgCyl1WrjigkZK7i4pLNjgIAGWJX8d27d68aN24sSVqyZIkKFy6s06dPa/bs2friiy8cGhAAkH18991+dehw84ps7dsvVGJiitmRACDd7Cq+cXFx8vX1lSRt2LBBHTp0kIuLi+rVq6fTp087NCAAIHuYOnWPevRYptTUm1fvLFQor1xcLPd5FABkH3YV37Jly2rFihU6e/as1q9fb1vXe+nSJfn5cVIDAOQ248bt0Esvfa9bV6x/6aWamj27vdzcXM0NBgAZYFfxHT58uIYOHapSpUqpTp06ql+/vqSbs7/Vq1d3aEAAgHkMw9DIkZs1dGiYbWzo0PqaNKkVs70Achy7dnXo2LGjGjVqpAsXLqhq1aq28WbNmql9+/YOCwcAMI9hGHrjjQ367LNdtrEPPnhS7733RJodHQAgp7D7AhZFihRRkSJFdO7cOUlSiRIlVKdOHYcFAwCYJzXVqv79v9c33/xmG/vssxC9/no9E1MBwIOxa6mD1WrVBx98IH9/f5UsWVIlS5ZUvnz5NHr0aFmtVkdnBABksdRUw3YJYotF+uab1pReADmeXTO+7777rr799luNHTtWDRs2lCRt375dI0eOVEJCgj788EOHhgQAZC13d1ctXdpZbdosUN++1dWlS2WzIwHAA7MYxq1zdNOvWLFimjx5stq0aZNmfOXKlRo4cKDOnz/vsICOFh0dLX9/f0VFRbEDBQDcxz+v0IY7mFJCijkv+RSXXjpndhogV8isvmbXUofIyEhVqFDhtvEKFSooMjLygUMBALLWtWvx6tx5sc6ejUozTukFkJvYVXyrVq2qr7766rbxr776Ks0uDwCA7O/ixRg9+eQsLV58UEFBc3TxYozZkQAgU9i1xveTTz5Rq1attHHjRtsevjt37tTZs2e1Zs0ahwYEAGSes2ejFBQ0R0ePXpUkXb+eoMuX41S4sI/JyQDA8eya8W3SpImOHj2q9u3b6/r167p+/bo6dOigI0eOqHHjxo7OCADIBMeORapx4xm20hsY6Kdt2/qocuUAk5MBQObI8IxvcnKyWrRoocmTJ7N7AwDkUAcOXFJw8BxFRNxc1lC2bAFt3NhTJUvmMzcYAGSiDBdfNzc3/fHHH5mRBQCQBX799S+FhMxVZGS8JKly5QBt2PCsihb1NTkZAGQuu5Y6PPvss/r2228dnQUAkMm2bTutp56aZSu9tWsX0+bNz1F6ATgFu05uS0lJ0fTp07Vx40bVrFlT3t7eae4fP368Q8IBABxr/frjunEjSZL0xBMltXp1N/n5eZicCgCyhl3F98CBA6pRo4Yk6ejRo2nuY89HAMi+Ro9uqqtX43TqVJSWLu2svHndzI4EAFnGruK7adMmR+cAAGQBi8WiiRNbKSXFKnd3V7PjAECWsmuNLwAgZ5g8+Vf99NOZNGMuLhZKLwCnlO4Z3w4dOmjmzJny8/NThw4d7nnssmXLHjgYAODBjB27XcOGhcvf30M//vicatQoanYkADBVuouvv7+/bf2uv79/pgUCADwYwzD07rs/6qOPtkuSoqIStWHDcYovAKeX7uI7Y8aMO/4ZAJB9WK2GXnttrb766hfb2NixzfTWW41MTAUA2YPda3xTUlK0ceNGTZkyRTdu3JAk/fXXX4qJiXFYOABA+qWkWPXCC6vSlN6JE0MpvQDw/+za1eH06dNq0aKFzpw5o8TERAUHB8vX11cff/yxEhMTNXnyZEfnBADcQ1JSqnr0WKYlSw5KunkC28yZbdWzZ1WTkwFA9mHXjO9rr72mWrVq6dq1a/Ly8rKNt2/fXuHh4Q4LBwC4v7i4ZLVtu8BWet3cXLR4cSdKLwD8g10zvtu2bdOOHTvk7u6eZrxUqVI6f/68Q4IBANJn69bTWr/+mCTJyyuPli/vopCQsianysWOLJZ2DJeSbi7zU+wFc/MASDe7ZnytVqtSU1NvGz937px8fbneOwBkpRYtymrSpFby8/PQ+vXPUnoz247hUuRhKeb8zQ/DenPcnZ9/QHZnV/Ft3ry5JkyYYLttsVgUExOjESNGKDQ01FHZAADp9NJLtfTnn6+oceOSZkfJ/W7N9FpcJJ/iNz8KVJAajjY3F4D7smupw7hx4xQSEqJKlSopISFB3bt3159//qmCBQvqu+++c3RGAMDfnD59Xbt2nVOXLpXTjAcEeJuUyEl5F5VeOmd2CgAZYFfxLVGihH7//XctWLBAf/zxh2JiYvTCCy+oR48eaU52AwA41pEjVxQUNEd//XVDFotFnTs/ZnYkAMgx7Cq+kpQnTx49++yzjswCALiH33+PUHDwHF2+HCdJGj16qzp0qKg8eezekh0AnEq6i++qVavS/aRt2rSxKwwA4M527Tqnli3n6fr1BElS1aqFtWFDT0ovAGRAuotvu3bt0ty2WCwyDOO2MUl33PEBAGCfH388qTZtvlNsbLIkqV69Elqzprvy52dpGQBkRLqnCqxWq+1jw4YNqlatmtauXavr16/r+vXrWrt2rWrUqKF169ZlZl4AcCrff39UoaHzbKX3qadKKyysJ6UXAOxg1xrf119/XZMnT1ajRv+7/ntISIjy5s2rF198UYcOHXJYQABwVgsXHtCzzy5XSsrNfWJbty6vRYs6ydPT7tMzAMCp2bU47Pjx48qXL99t4/7+/jp16tQDRgIAXLsWr/79f7CV3q5dK2vp0s6UXgB4AHYV39q1a2vIkCG6ePGibezixYv617/+pTp16jgsHAA4q/z5vbRyZVd5euZR377VNXdue7m5uZodCwByNLumDqZPn6727dvr4YcfVmBgoCTp7NmzKleunFasWOHIfADgtJ54oqT27HlRFSsWtJ08DACwn13Ft2zZsvrjjz8UFhamw4cPS5IqVqyooKAg/nIGADsYhqGVK4+obdtH0/w9WqlSIRNTAUDuYvdiMYvFoubNm6t58+aOzAMATic11aqBA3/Q1Kl79e67jfXvfz9ldiQAyJXsLr6xsbHasmWLzpw5o6SkpDT3vfrqqw8cDACcQXJyqnr3Xqn58/dLksaM2aZOnSqpatUiJicDgNzHruL722+/KTQ0VHFxcYqNjVWBAgV05coV5c2bVwEBARRfAEiHhIQUde26RCtXHpEk5cnjotmz21F6ASCT2LWrw+DBg9W6dWtdu3ZNXl5e2rVrl06fPq2aNWvq008/dXRGAMh1YmOT1Lr1d7bS6+HhqmXLOqtbtyomJwOA3Muu4rtv3z698cYbcnFxkaurqxITExUYGKhPPvlE77zzjqMzAkCucv16gpo3n6uNG09Ikry93fTDD93VuvWjJicDgNzNruLr5uYmF5ebDw0ICNCZM2ck3byAxdmzZx2XDgBymcuXY9W06Szt2HHz78p8+TwVFtZTzZqVMTkZAOR+dq3xrV69un755ReVK1dOTZo00fDhw3XlyhXN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Sk5OTnPc9evXNWjQIBUtWlQeHh4qX7681qxZY7v/vffe04cffqioqKjM+LTsxsltDrZr1zm1bDlP168nSJKqVi2sDRt6KiDA2+RkAIDc5NSpU2rYsKHy5cun//znP6pSpYqSk5O1fv16DRo0SIcPH7breQ3DUGpqqvLkybyKsHHjRj322GO6cuWKPvzwQz399NM6evSoChcuLOlm6a1Xr57c3d01efJkPfbYYzp16pTee+891a5dWzt37lSZMmUk2f8+GIahr776Sh988MFt93l6emr37t3p+lySkpLk7u5u5zuRuU6ePKlWrVqpf//+mjdvnsLDw9W3b18VLVpUISEhd3yMm5ubevXqpRo1aihfvnz6/fff1a9fP1mtVo0ZM0bSzc85ODhYAQEBWrJkiYoXL67Tp08rX758tuepXLmyHnnkEc2dO1eDBg3Kik83fQwnExUVZUgyoqKiHP7c4eEnDG/vDw1ppCGNNOrX/8aIjIz73wGTixvGp7r5XwBAthAfH28cPHjQiI+PNztKhrRs2dIoXry4ERMTc9t9165dMwzDME6ePGlIMn777bc090kyNm3aZBiGYWzatMmQZKxZs8aoUaOG4ebmZkyZMsWQZBw6dCjN844fP94oU6aMYRiGkZKSYjz//PNGqVKlDE9PT6N8+fLGhAkT7pn5Tnn++OMPQ5KxcuVK21j//v0Nb29v48KFC2keHxcXZxQvXtxo0aJFht6HO/nll18MFxcXIzo6Os34m2++aZQrV87w8vIySpcubbz33ntGUlKS7f4RI0YYVatWNaZNm2aUKlXKsFgsttd64YUXjIIFCxq+vr5G06ZNjX379tked+zYMaNNmzZGQECA4e3tbdSqVcsICwu75/v1oN58803jscceSzPWpUsXIyQkJEPPM3jwYKNRo0a225MmTTLKlCmT5n25k1GjRqV53D/d63svs/oav3x3kO+/P6rQ0HmKjb35q4CnniqtDRt6Kn9+L5OTAQBym8jISK1bt06DBg2St/ftv1H8+8xber399tsaO3asDh06pI4dO6pWrVqaN29emmPmzZun7t27S7q5/3GJEiW0ePFiHTx4UMOHD9c777yjRYsWpfs14+PjNXv2bEmyzZparVYtWLBAPXr0UJEiRdIc7+XlpYEDB2r9+vWKjIx8oPdh27ZtKl++vHx9fdOM+/r6aubMmTp48KC++OILffvtt/rss8/SHHPs2DEtXbpUy5Yt0759+yRJnTp10qVLl7R27Vrt2bNHNWrUULNmzRQZGSlJiomJUWhoqMLDw/Xbb7+pRYsWat269R2Xbvw9o4+Pzz0//vk1+rudO3cqKCgozVhISIh27tx518f807Fjx7Ru3To1adLENrZq1SrVr19fgwYNUuHChVW5cmWNGTPmtqse1qlTR7t371ZiYmK6Xy+zsdTBAS5dilWXLkuUmHjzC966dXktWtRJnp68vQCQY82tJcVGZO1reheRnv31vocdO3ZMhmGoQoUKDnvpDz74QMHBwbbbPXr00FdffaXRo0dLko4ePao9e/Zo7ty5km7+SnzUqFG240uXLq2dO3dq0aJF6ty58z1fq0GDBnJxcVFcXJwMw1DNmjXVrFkzSdLly5d1/fr1u66trVixogzD0LFjxyTJ7vfh9OnTKlas2G3j7733nu3PpUqV0htvvKHvvvtOb775pm08KSlJs2fPVqFChSRJ27dv1+7du3Xp0iV5eHhIkj799FOtWLFCS5Ys0YsvvqiqVauqatWqtucYPXq0li9frlWrVunll1++Y8ZatWrZivXd3FoecicRERG33V+4cGFFR0crPj5eXl53n5xr0KCB9u7dq8TERL344otploScOHFCP/74o3r06KE1a9bo2LFjGjhwoJKTkzVixAjbccWKFVNSUpIiIiJUsmTJe34eWYVm5gABAd6aMaOtunVbqs6dH9Ps2e3k5pa1154GADhYbIQUc97sFHdkGIbDn7NWrVppbnft2lVDhw7Vrl27VK9ePc2bN081atRIUzInTpyo6dOn68yZM4qPj1dSUpKqVat239dauHChKlSooAMHDujNN9/UzJkz5ebmluaY9HyOD/I+xMfHy9PT87bxWbNm6bPPPtOxY8cUGxsrSbaCe0vJkiXTjP3++++KiYnRQw89dNtrHD9+XNLNGd+RI0fqhx9+0IULF5SSkqL4+Ph7zvh6eXmpbFlzrvC6cOFC3bhxQ7///rv+9a9/6dNPP7WVf6vVqoCAAE2dOlWurq6qWbOmzp8/r//85z9piu+tYp2dLgVO8XWQzp0fU7Fivqpfv4RcXVlBAgA5nneR+x9j0muWK1dOFovlview3boox98L4j/Pzre99D+WChQpUkRPPfWU5s+fr3r16mn+/PkaMGCA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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Tem o objetivo visualizar a curva ROC (Receiver Operating Characteristic) com informações sobre a área sob a curva (AUC-ROC) para avaliar de forma gráfica o desempenho de um modelo de classificação binária.\n",
        "\n",
        "Esse gráfico fornece uma representação visual do desempenho do modelo, ajudando a avaliar sua capacidade de distinguir entre as classes positivas e negativas em diferentes limiares de decisão."
      ],
      "metadata": {
        "id": "kGw2NTHaPJEb"
      }
    }
  ]
}