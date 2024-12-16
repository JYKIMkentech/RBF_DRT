clc; clear; close all;

%% 설정
axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

lambda_grids = logspace(-4, 2, 5);
num_lambdas = length(lambda_grids);
OCV = 0;
R0 = 0.1;

%% RBF 파라미터 설정
rbf_type = 'gaussian';
shape_param = 1.0;

%% 데이터 로드
save_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_lambda\';
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_lambda\';
mat_files = dir(fullfile(file_path, '*.mat'));
if isempty(mat_files)
    error('데이터 파일이 존재하지 않습니다. 경로를 확인해주세요.');
end
for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% 데이터셋 선택
datasets = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};
disp('데이터셋을 선택하세요:');
for i = 1:length(datasets)
    fprintf('%d. %s\n', i, datasets{i});
end
dataset_idx = input('데이터셋 번호를 입력하세요: ');
if isempty(dataset_idx) || dataset_idx < 1 || dataset_idx > length(datasets)
    error('유효한 데이터셋 번호를 입력해주세요.');
end
selected_dataset_name = datasets{dataset_idx};
if ~exist(selected_dataset_name, 'var')
    error('선택한 데이터셋이 로드되지 않았습니다.');
end
selected_dataset = eval(selected_dataset_name);

%% 타입 선택 및 데이터 준비
types = unique({selected_dataset.type});
disp('타입을 선택하세요:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('타입 번호를 입력하세요: ');
if isempty(type_idx) || type_idx < 1 || type_idx > length(types)
    error('유효한 타입 번호를 입력해주세요.');
end
selected_type = types{type_idx};
type_indices = strcmp({selected_dataset.type}, selected_type);
type_data = selected_dataset(type_indices);
if isempty(type_data)
    error('선택한 타입에 해당하는 데이터가 없습니다.');
end
SN_list = [type_data.SN];

%% 새로운 필드 추가 
new_fields = {'Lambda_vec', 'CVE', 'Lambda_hat'};
num_elements = length(selected_dataset);
empty_fields = repmat({[]}, 1, num_elements);

for nf = 1:length(new_fields)
    field_name = new_fields{nf};
    if ~isfield(selected_dataset, field_name)
        [selected_dataset.(field_name)] = empty_fields{:};
    end
end

%% 람다 최적화 및 교차 검증
scenario_numbers = SN_list; 
validation_combinations = nchoosek(scenario_numbers, 2); 
num_folds = size(validation_combinations, 1); 
CVE_total = zeros(num_lambdas,1); 

for m = 1 : num_lambdas
    lambda = lambda_grids(m);
    CVE = 0 ;

    for f = 1 : num_folds
        val_trips = validation_combinations(f,:);
        train_trips = setdiff(1 : length(type_data), val_trips);

        W_total = [];
        y_total = [];

        % 학습 데이터셋에 대해 W_total, y_total 구하기 (RBF기반)
        for s = train_trips
            t = type_data(s).t;
            dt = [t(1); diff(t)];
            dur = type_data(s).dur;
            n = type_data(s).n;
            I = type_data(s).I;
            V = type_data(s).V;

            % RBF 기반 DRT estimation 호출
            [~, ~, ~, ~, W, y] = DRT_estimation_RBF(t, I, V, lambda, n, dt, dur, OCV, R0, rbf_type, shape_param);

            W_total = [W_total; W]; 
            y_total = [y_total; y]; 
        end

        % 학습 데이터로 gamma_total(x_coeff) 구하기
        [gamma_total] = DRT_estimation_with_Wy(W_total, y_total, lambda);

        % 검증 데이터셋에 대해 CVE 계산
        for j = val_trips
            t = type_data(j).t;
            dt = [t(1); diff(t)];
            dur = type_data(j).dur;
            n = type_data(j).n;
            I = type_data(j).I;
            V = type_data(j).V;

            [~, ~, ~, ~, W_val, ~] = DRT_estimation_RBF(t, I, V, lambda, n, dt, dur, OCV, R0, rbf_type, shape_param);
            V_est = OCV + I * R0 + W_val * gamma_total;

            error = sum((V - V_est).^2);
            CVE = CVE + error;
        end
    end

    CVE_total(m) = CVE;
    fprintf('Lambda: %.2e, CVE: %.4f\n', lambda, CVE_total(m));
end

[~, optimal_idx] = min(CVE_total);
optimal_lambda = lambda_grids(optimal_idx);

%% 결과 저장
for i = 1:length(type_data)
    type_data(i).Lambda_vec = lambda_grids;
    type_data(i).CVE = CVE_total;
    type_data(i).Lambda_hat = optimal_lambda;
end
selected_dataset(type_indices) = type_data;

% % 폴더가 존재하지 않으면 생성
% if ~exist(save_path, 'dir')
%     mkdir(save_path);
% end
% save(fullfile(save_path, [selected_dataset_name, '.mat']), selected_dataset_name);
% fprintf('Updated dataset saved to %s\n', fullfile(save_path, [selected_dataset_name, '.mat']));

%% Plot (CVE vs lambda)
figure;
semilogx(lambda_grids, CVE_total, 'b-', 'LineWidth', 1.5); hold on;
semilogx(optimal_lambda, CVE_total(optimal_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\lambda', 'FontSize', labelFontSize);
ylabel('CVE', 'FontSize', labelFontSize);
title('CVE vs \lambda ', 'FontSize', titleFontSize);
grid on;
legend({'CVE', ['Optimal \lambda = ', num2str(optimal_lambda, '%.2e')]}, 'Location', 'best');
hold off;

%% function 
function [gamma_total] = DRT_estimation_with_Wy(W_total, y_total, lambda)
    W_total_n = size(W_total, 2);

    L = zeros(W_total_n-1, W_total_n);
    for i = 1:W_total_n-1
        L(i, i) = -1;
        L(i, i+1) = 1;
    end

    H = 2 * (W_total' * W_total + lambda * (L' * L));
    f = -2 * W_total' * y_total;

    A_ineq = -eye(W_total_n);
    b_ineq = zeros(W_total_n , 1);

    options = optimoptions('quadprog', 'Display', 'off');
    gamma_total = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);  
end
