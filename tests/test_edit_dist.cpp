#include <vector>
#include <iostream>

int edit_distance(const std::vector<int> &A, const std::vector<int> &B) {

    std::vector<std::vector<int>> dist{{},
                                       {}};
    int pre = 0, cur = 1;
    auto max_len = std::max(A.size(), B.size());
    for (int j = 0; j < max_len + 10; ++j) {
        dist[pre].push_back(j); // 从 空字符串 变换到 B_j
        dist[cur].push_back(0); // 初始化
    }
    for (int i = 1; i <= A.size(); ++i) {
        dist[cur][0] = i;
        for (int j = 1; j <= B.size(); ++j) {
            dist[cur][j] = std::max(i, j); // i直接到j的最大距离是max(i,j)

            dist[cur][j] = std::min(dist[cur][j], dist[pre][j] + 1); // 删除A[i]
            dist[cur][j] =
                    std::min(dist[cur][j], dist[cur][j - 1] + 1); // 删除B[j]
            dist[cur][j] =
                    std::min(dist[cur][j],
                             dist[pre][j - 1] + (A[i - 1] == B[j - 1] ? 0 : 1));
            // A_i != B_j , +1 替换; A_i == B_j , +0;
        }
        std::cout << i<<':';
        for(int j = 0; j<=B.size(); ++j){
            std::cout << dist[cur][j] <<' ';
        }
        std::cout<<'\n';
        std::swap(pre, cur);
    }
    return dist[pre][B.size()];
}

/**
 * sfdqxbw
 * gfdgw
 *
 * 4
 * @return
 */


int main() {
    std::vector<int> A{115, 102, 100, 113, 120, 98, 119},
            B{103, 102, 100, 103, 119};
    auto x = edit_distance(A, B);
    std::cout << x << '\n';
    A.clear();
    B.clear();
    for (int i = 1; i <= 20; ++i) {
        A.push_back(i);
    }
    for (int i = 1; i <= 4; ++i) {
        B.push_back(i);
    }
    std::cout << edit_distance(A, B) << '\n';
    return 0;
}