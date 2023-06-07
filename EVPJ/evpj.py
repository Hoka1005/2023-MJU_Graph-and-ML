import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import csv
import copy

from google.colab import drive
drive.mount('/content/gdrive')

##################################################
# Change FILE_NAME for test algorithm
##################################################
FILE_PATH = 'gdrive/MyDrive/Colab_Notebooks/EVPJ/'
FILE_NAME = FILE_PATH + 'EVPJ_Example_data_2.csv'
##################################################

Node_A = []
Node_B = []
total = []
need = []
original_graph = []
complete_graph_list = []
fixed_edge_list = []



def possible_node_list_in_Node_A(node, need_n, complete_n):
    possible_node_list = []

    print("Node_A : " + str(Node_A))

    start_index = Node_A.index(node) 

    for current_node in Node_A[start_index:]:
        index = Node_A.index(current_node) 
        if need_n[index] > 0 and complete_n[index] == False :
            possible_node_list.append(current_node)

    print("possiblie_node_list_of_A : " + str(possible_node_list) + "\n")
    return possible_node_list



def possible_target_list_in_Node_B(node, target, selected_n, left_n):
    possible_target_list = []

    start_index = 0

    for current_target in Node_B[start_index:]:
        j = Node_B.index(current_target) 
        edge = (node, current_target)
        if edge in original_graph :
            print("edge"+ str(edge) + " is in original_graph --> pass")
            if edge not in selected_n : 
                print("edge not in selected --> pass")
                if left_n[j] > 0:
                    print("left[" + str(j) + "] : " + str(left_n[j]) + " > 0 --> pass")
                    possible_target_list.append(current_target)
                else : 
                    print("left[" + str(j) + "] : " + str(left_n[j]) + " <= 0 --> fail")
            else : 
                print("edge in selected --> fail")
        else:
            print("edge"+ str(edge) + " is not in original_graph --> fail")
            
    print("\npossiblie_target_list_of_B : " + str(possible_target_list))
    return possible_target_list



def check_complete(complete_c, complete_graph_c):
    global total_num
    print("\n-----| Checking Complete Graph...")
    
    is_complete = True
    
    for i in range(len(Node_A)):
        if not complete_c[i]:
            is_complete = False
    
    if is_complete:
        if complete_graph_c not in complete_graph_list:
            print("-----| Complete Graph Found!")
            print("-----| complete_graph : " + str(complete_graph_c) + "\n")
            complete_graph_list.append(complete_graph_c)
            total_num += 1
        else: 
            print("-----| Complete Graph Already Exists!\n")

    else:
        print("-----| Not Complete Graph...\n")



def select_edge(node, target, need_n, left_n, complete_n, selected_n):
    edge = (node, target)
    
    need_n_cp = copy.deepcopy(need_n)
    complete_n_cp = copy.deepcopy(complete_n)
    left_n_cp = copy.deepcopy(left_n)
    selected_n_cp = copy.deepcopy(selected_n)
    
    print("\n=== The edge : " + str(edge) + " ===")
    print("need_n_cp : " + str(need_n_cp))
    print("left_n_cp : " + str(left_n_cp))
    print("complete_n_cp : " + str(complete_n_cp))
    print("selected_n_cp : " + str(selected_n_cp))
    
    if need_n_cp[Node_A.index(node)] > 0 and left_n_cp[Node_B.index(target)] > 0 and edge not in selected_n_cp:
        selected_n_cp.append(edge)
        need_n_cp[Node_A.index(node)] -= 1
        left_n_cp[Node_B.index(target)] -= 1
        if need_n_cp[Node_A.index(node)] == 0:
            complete_n_cp[Node_A.index(node)] = True
        
        print("\nValid Edge, Selecting...")
        print("need_n_cp[" + str(Node_A.index(node)) + "] : " + str(need_n_cp[Node_A.index(node)]) + 
              ",   left_n_cp[" + str(Node_B.index(target)) + "] : " + str(left_n_cp[Node_B.index(target)]) +
              ",   complete_n_cp[" + str(Node_A.index(node)) + "] : " + str(complete_n_cp[Node_A.index(node)]) + "\n")
        
        fixed_edge_list_cp = copy.deepcopy(fixed_edge_list)
        complete_n_graph = fixed_edge_list_cp + selected_n_cp
        complete_n_graph.sort(key=lambda x: (x[0], x[1]))
        
        print("fixed_edge_list_cp : " + str(fixed_edge_list_cp) )
        print("selected_n_cp : " + str(selected_n_cp) )
        print("complete_n_graph : " + str(complete_n_graph) )
        
        check_complete(complete_n_cp, complete_n_graph)
        
        for n in possible_node_list_in_Node_A(node, need_n_cp, complete_n_cp):
            for t in possible_target_list_in_Node_B(n, target, selected_n_cp, left_n_cp):
                select_edge(n, t, need_n_cp, left_n_cp, complete_n_cp, selected_n_cp)
        
    else:
        print("\nInvalid Edge, Not Selecting... \n")
        
        

# Main function
def main():

    # Initialize variables
    selected = []
    global total_num
    total_num = 0
    N = 2   # Set maximum edge number of right node 
    global Node_A, Node_B, original_graph, total, need, complete, left, complete_graph_list, fixed_edge_list
    
    # Read original graph information form CSV
    df = pd.read_csv(FILE_NAME)
    print("DF\n" + str(df))

    Node_A = df.iloc[:, 0].tolist()
    print("Node_A : " + str(Node_A))

    time_scope = range( min(df.iloc[:, 1].tolist() + df.iloc[:, 2].tolist()), max(df.iloc[:, 1].tolist() + df.iloc[:, 2].tolist()) + 1 )
    #print("Time Scope : " + str(time_scope))
    for i in time_scope[:]:
        Node_B.append(str(i))
    print("Node_B : " + str(Node_B))

    for _, row in df.iterrows():
        left_node = row[0]
        start_node = int(row[1])
        end_node = int(row[2])
        for right_node in range(start_node, end_node+1):
            original_graph.append( (left_node, str(right_node)) )

    print("original_graph : " + str(original_graph))

    total = (df.iloc[:, 2] - df.iloc[:, 1] + 1).tolist()
    need = df.iloc[:, 3].tolist()
    complete = [False] * len(Node_A)
    left = [N] * len(Node_B)

    # Generate Bipartite Graph with NetworkX
    G = nx.Graph()
    G.add_nodes_from(Node_A, bipartite=0)
    G.add_nodes_from(Node_B, bipartite=1)
    G.add_edges_from(original_graph)

    # Node Information
    print("----------------------------------")
    print("total : " + str(total))
    print("need  : " + str(need))
    print("complete : " + str(complete))
    print("left : " + str(left))

    # Initial Validating
    for node in Node_A:
        if sum(total) < sum(need):
            print("Invalid Input: Invalid Node Exist")
            print("sum(total) : " + str(sum(total)))
            print("sum(need)  : " + str(sum(need)))
            return

    if sum(need) > N * len(Node_B):
        print("Invalid Input: Total needed time exceeds the maximum available charging time")
        print("Total needed time : " + str(sum(total)))
        print("maximum available charging time : " + str(N * len(Node_B)))
        return
                    
                    
    # Set fixed edge list
    print("----------------------------------")
    print("Set fixed edge list")
    for i, node in enumerate(Node_A):
        print("i : " + str(i))
        print("node : " + str(node))
        print("total[i] : " + str(total[i]))
        print("need[i] : " + str(need[i]))
        if total[i] == need[i]:
            complete[i] = True
            need[i] = 0
            for j, target in enumerate(Node_B):
                edge = (node, target)
                if edge in original_graph:
                    print("Fixed Edge append : ")
                    print(edge)
                    fixed_edge_list.append(edge)
                    left[j] -= 1
    print("fixed_edge_list : " + str(fixed_edge_list))
    
    # Check if complete graph is found
    is_complete = True
    for i in range(len(Node_A)):
        if not complete[i]:
            is_complete = False
    if is_complete:
        print("-----| Complete Graph Found!")
        print("-----| complete_graph : " + str(original_graph))
        complete_graph_list.append(original_graph)
        total_num += 1
        
    # Get first node and target
    else:
        # Start DFS
        print("----------------------------------")
        print("Start")
        
        for n in possible_node_list_in_Node_A(Node_A[0], need, complete):
            for t in possible_target_list_in_Node_B(n, Node_B[0], selected, left):
                select_edge(n, t, need, left, complete, selected)

    # Print Result
    print("----------------------------------")
    print("Complete Graph List: ")
    for complete_graph in complete_graph_list:
        print(complete_graph)
    print("Total Number of Complete Graph: " + str(total_num))

    if total_num <= 0:
        print("Not available for scheduling")

    else: 
        # Set Node Position
        pos = {}
        for i, node in enumerate(Node_A):
            pos[node] = (0, 2 * (len(Node_A)-i-1) )
        for i, node in enumerate(Node_B):
            if node not in Node_A:
                pos[node] = (1, 2 * (len(Node_B)-i-1) )

        # Draw Bipartite Graph with NetworkX
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_color='black')
        nx.draw_networkx_edges(G, pos, edgelist=complete_graph_list[0], edge_color='red', width=2)
        plt.show()



if __name__ == "__main__":
    main()
    drive.flush_and_unmount()
